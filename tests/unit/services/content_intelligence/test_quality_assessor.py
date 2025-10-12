"""Unit tests for the quality assessor service."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from ._dependency_stubs import load_content_intelligence_module


if TYPE_CHECKING:  # pragma: no cover
    from src.services.content_intelligence.models import QualityScore
    from src.services.content_intelligence.quality_assessor import QualityAssessor


@pytest.fixture(scope="module")
def quality_assessor_types() -> dict[str, Any]:
    """Import quality assessor types once dependencies are stubbed.

    Returns:
        dict[str, Any]: Mapping of quality assessor support types for the test suite.
    """

    assessor_module = load_content_intelligence_module("quality_assessor")
    models_module = load_content_intelligence_module("models")
    return {
        "assessor_cls": assessor_module.QualityAssessor,
        "score_cls": models_module.QualityScore,
    }


@pytest.fixture(scope="module")
def assessor_cls(quality_assessor_types: dict[str, Any]) -> type[QualityAssessor]:
    """Expose QualityAssessor class for fixtures.

    Args:
        quality_assessor_types: Mapping of quality assessor support types.

    Returns:
        type[QualityAssessor]: Quality assessor class reference.
    """

    return quality_assessor_types["assessor_cls"]


@pytest.fixture(scope="module")
def score_cls(quality_assessor_types: dict[str, Any]) -> type[QualityScore]:
    """Expose QualityScore dataclass for assertions.

    Args:
        quality_assessor_types: Mapping of quality assessor support types.

    Returns:
        type[QualityScore]: Dataclass type used to represent assessment outputs.
    """

    return quality_assessor_types["score_cls"]


def _text_to_vector(text: str) -> list[float]:
    """Convert text into a lightweight numeric embedding for tests.

    Args:
        text: Raw text that should be converted into a numeric vector.

    Returns:
        list[float]: Simplified numeric embedding representation.
    """

    tokens = text.lower().split()
    unique_tokens = set(tokens)
    return [
        float(len(tokens)),
        float(len(unique_tokens)),
        float(sum(ord(char) for char in text) % 1000) or 1.0,
    ]


@pytest.fixture
def embedding_manager() -> AsyncMock:
    """Create an embedding manager stub for quality assessments.

    Returns:
        AsyncMock: Stubbed embedding manager used in quality tests.
    """

    async def _generate_embeddings(**kwargs: Any) -> dict[str, Any]:
        texts: list[str] = list(kwargs.get("texts", []))
        return {
            "success": True,
            "embeddings": [_text_to_vector(text) for text in texts],
        }

    manager = AsyncMock()
    manager.generate_embeddings = AsyncMock(side_effect=_generate_embeddings)
    return manager


@pytest.fixture
def empty_manager_factory(
    assessor_cls: type[QualityAssessor],
) -> Callable[[], QualityAssessor]:
    """Create a factory returning assessors without embedding support.

    Args:
        assessor_cls: Quality assessor class reference.

    Returns:
        Callable[[], QualityAssessor]: Factory producing assessors lacking embeddings.
    """

    def _factory() -> QualityAssessor:
        return assessor_cls(embedding_manager=None)

    return _factory


@pytest.fixture
async def quality_assessor(
    assessor_cls: type[QualityAssessor], embedding_manager: AsyncMock
) -> QualityAssessor:
    """Provide an initialized QualityAssessor for tests.

    Args:
        assessor_cls: Quality assessor class reference.
        embedding_manager: Stubbed embedding manager fixture.

    Returns:
        QualityAssessor: Initialized quality assessor instance for tests.
    """

    assessor = assessor_cls(embedding_manager=embedding_manager)
    await assessor.initialize()
    return assessor


class TestQualityAssessor:
    """Validate composite scoring and helper utilities in QualityAssessor."""

    @pytest.mark.asyncio
    async def test_assess_quality_returns_composite_score(
        self, quality_assessor: QualityAssessor, score_cls: type[QualityScore]
    ) -> None:
        """Assessing quality yields a populated QualityScore dataclass.

        Args:
            quality_assessor: Initialized quality assessor instance.
            score_cls: Dataclass type used to represent assessment outputs.
        """

        content = (
            "Comprehensive documentation with examples, tutorials, and updated "
            "references that users can follow easily."
        )
        metadata = {"success": True, "quality_score": 0.8, "provider": "crawl4ai"}

        score = await quality_assessor.assess_quality(
            content=content,
            confidence_threshold=0.6,
            query_context="API documentation",
            extraction_metadata=metadata,
            existing_content=["Older documentation with fewer examples."],
        )

        assert isinstance(score, score_cls)
        assert 0.0 <= score.overall_score <= 1.0
        assert score.meets_threshold is True
        assert score.completeness >= 0.0
        assert score.duplicate_similarity >= 0.0

    @pytest.mark.asyncio
    async def test_duplicate_similarity_uses_embeddings(
        self, quality_assessor: QualityAssessor
    ) -> None:
        """Duplicate detection should surface high similarity for matching content.

        Args:
            quality_assessor: Initialized quality assessor instance.
        """

        content = "Frequently asked questions about account security."
        existing = [
            "Frequently asked questions about account security and recovery options.",
            "Unrelated announcement about product pricing.",
        ]

        similarity = await quality_assessor._assess_duplicate_similarity(  # pylint: disable=protected-access
            content=content, existing_content=existing
        )

        assert similarity > 0.8

    @pytest.mark.asyncio
    async def test_relevance_scoring_uses_embeddings(
        self, quality_assessor: QualityAssessor
    ) -> None:
        """Embedding-aware relevance scoring should approach perfect similarity.

        Args:
            quality_assessor: Initialized quality assessor instance.
        """

        content = "Step-by-step tutorial for configuring webhooks."
        relevance = await quality_assessor._assess_relevance(  # pylint: disable=protected-access
            content=content,
            query_context="Step-by-step tutorial for configuring webhooks",
        )

        assert relevance == pytest.approx(1.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_quality_assessor_handles_missing_embeddings(
        self,
        empty_manager_factory: Callable[[], QualityAssessor],
        score_cls: type[QualityScore],
    ) -> None:
        """Assessment still succeeds when embeddings are unavailable.

        Args:
            empty_manager_factory: Factory returning assessors without embeddings.
            score_cls: Dataclass type used to represent assessment outputs.
        """

        assessor = empty_manager_factory()
        await assessor.initialize()

        score = await assessor.assess_quality(
            content="Short announcement without semantic metadata.",
            confidence_threshold=0.2,
            query_context=None,
            extraction_metadata=None,
            existing_content=None,
        )

        assert isinstance(score, score_cls)
        assert score.meets_threshold is True
