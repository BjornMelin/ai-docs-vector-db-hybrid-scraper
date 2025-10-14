"""Unit tests for the content classifier service."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from ._dependency_stubs import load_content_intelligence_module


if TYPE_CHECKING:  # pragma: no cover
    from src.services.content_intelligence.classifiers import ContentClassifier
    from src.services.content_intelligence.models import (
        ContentClassification,
        ContentType,
    )


@pytest.fixture(scope="module")
def classifier_types() -> dict[str, Any]:
    """Import classifier types with external dependencies stubbed.

    Returns:
        dict[str, Any]: Mapping of classifier support types for the test suite.
    """
    classifier_module = load_content_intelligence_module("classifiers")
    models_module = load_content_intelligence_module("models")
    return {
        "classifier_cls": classifier_module.ContentClassifier,
        "classification_cls": models_module.ContentClassification,
        "content_type_enum": models_module.ContentType,
    }


@pytest.fixture(scope="module")
def content_type_enum(classifier_types: dict[str, Any]) -> ContentType:
    """Expose ContentType enum for type-aware assertions.

    Args:
        classifier_types: Mapping containing classifier-related types.

    Returns:
        ContentType: Enumeration representing classified content categories.
    """
    return classifier_types["content_type_enum"]


@pytest.fixture(scope="module")
def classification_cls(
    classifier_types: dict[str, Any],
) -> type[ContentClassification]:
    """Expose ContentClassification dataclass for assertions.

    Args:
        classifier_types: Mapping containing classifier-related types.

    Returns:
        type[ContentClassification]: Dataclass type used for result validation.
    """
    return classifier_types["classification_cls"]


def _one_hot(index: int, size: int) -> list[float]:
    """Create a one-hot vector for deterministic embedding fixtures.

    Args:
        index: Position that should be assigned the ``1.0`` value.
        size: Total number of positions within the vector.

    Returns:
        list[float]: One-hot encoded vector with a deterministic pattern.
    """
    return [1.0 if position == index else 0.0 for position in range(size)]


@pytest.fixture
def deterministic_embedding_manager() -> AsyncMock:
    """Create an embedding manager that returns deterministic embeddings.

    Returns:
        AsyncMock: Stubbed embedding manager with deterministic responses.
    """
    reference_order = [
        "This is documentation that explains how to use a software system or API.",
        "This is source code with functions, classes, and programming logic.",
        "This is a frequently asked questions section with questions and answers.",
        "This is a step-by-step tutorial that teaches how to accomplish a task.",
        "This is an API reference with technical specifications and parameters.",
        "This is a blog post or article written by an author on a specific topic.",
        "This is a news article reporting on recent events or announcements.",
        "This is a forum discussion with posts, replies, and community interaction.",
    ]

    embeddings_by_reference = {
        text: _one_hot(index, len(reference_order))
        for index, text in enumerate(reference_order)
    }

    async def _generate_embeddings(**kwargs: Any) -> dict[str, Any]:
        texts: list[str] = list(kwargs.get("texts", []))
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "def " in lowered or "class " in lowered:
                vectors.append(_one_hot(1, len(reference_order)))
                continue
            if "documentation" in lowered or "api" in lowered:
                vectors.append(_one_hot(0, len(reference_order)))
                continue
            if "tutorial" in lowered or "step-by-step" in lowered:
                vectors.append(_one_hot(3, len(reference_order)))
                continue
            if text in embeddings_by_reference:
                vectors.append(embeddings_by_reference[text])
                continue
            vectors.append(_one_hot(5, len(reference_order)))
        return {"success": True, "embeddings": vectors}

    manager = AsyncMock()
    manager.generate_embeddings = AsyncMock(side_effect=_generate_embeddings)
    return manager


@pytest.fixture
def failing_classifier_factory(
    classifier_types: dict[str, Any],
) -> Callable[[], ContentClassifier]:
    """Create classifiers whose embedding managers always fail.

    Args:
        classifier_types: Mapping containing classifier-related types.

    Returns:
        Callable[[], ContentClassifier]: Factory producing classifiers with
            failing embedding managers.
    """
    classifier_cls = classifier_types["classifier_cls"]

    async def _generate_embeddings(**_kwargs: Any) -> dict[str, Any]:
        return {"success": False, "embeddings": []}

    def _factory() -> ContentClassifier:
        manager = AsyncMock()
        manager.generate_embeddings = AsyncMock(side_effect=_generate_embeddings)
        return classifier_cls(embedding_manager=manager)

    return _factory


@pytest.fixture
async def classifier(
    classifier_types: dict[str, Any],
    deterministic_embedding_manager: AsyncMock,
) -> ContentClassifier:
    """Provide an initialized classifier with deterministic embeddings.

    Args:
        classifier_types: Mapping containing classifier-related types.
        deterministic_embedding_manager: Stubbed embedding manager fixture.

    Returns:
        ContentClassifier: Initialized classifier instance for tests.
    """
    classifier_cls = classifier_types["classifier_cls"]
    instance = classifier_cls(embedding_manager=deterministic_embedding_manager)
    await instance.initialize()
    return instance


class TestContentClassifier:
    """Validate ContentClassifier behaviour and reasoning outputs."""

    @pytest.mark.asyncio
    async def test_classify_content_detects_code_blocks(
        self,
        classifier: ContentClassifier,
        classification_cls: type[ContentClassification],
        content_type_enum: ContentType,
    ) -> None:
        """Ensure code snippets are flagged correctly and typed as code.

        Args:
            classifier: Initialized classifier under test.
            classification_cls: Dataclass type for classification results.
            content_type_enum: Enumeration representing content types.
        """
        content = """

        def calculate_sum(numbers: list[int]) -> int:
            total = 0
            for value in numbers:
                total += value
            return total
        """.strip()

        result = await classifier.classify_content(
            content=content, url="https://example.com/src/module.py"
        )

        assert isinstance(result, classification_cls)
        assert result.primary_type == content_type_enum.CODE
        assert result.has_code_blocks is True
        assert "python" in result.programming_languages

    @pytest.mark.asyncio
    async def test_semantic_classification_scores_documentation(
        self,
        classifier: ContentClassifier,
        deterministic_embedding_manager: AsyncMock,
        content_type_enum: ContentType,
    ) -> None:
        """Verify semantic similarity contributes documentation confidence.

        Args:
            classifier: Initialized classifier under test.
            deterministic_embedding_manager: Embedding manager stub used for assertions.
            content_type_enum: Enumeration representing content types.
        """
        content = (
            "Comprehensive documentation guide describing API authentication flows."
        )

        result = await classifier.classify_content(
            content=content, url="https://example.com/docs/overview"
        )

        assert deterministic_embedding_manager.generate_embeddings.await_count == 1
        documentation_score = result.confidence_scores[content_type_enum.DOCUMENTATION]
        assert documentation_score >= 0.5
        assert (
            documentation_score >= result.confidence_scores[content_type_enum.REFERENCE]
        )
        assert result.primary_type == content_type_enum.DOCUMENTATION

    @pytest.mark.asyncio
    async def test_semantic_classification_handles_embedding_failure(
        self,
        failing_classifier_factory: Callable[[], ContentClassifier],
        classification_cls: type[ContentClassification],
    ) -> None:
        """Gracefully handle embedding failures by returning neutral scores.

        Args:
            failing_classifier_factory: Factory producing classifiers without
                embeddings.
            classification_cls: Dataclass type for classification results.
        """
        classifier_instance = failing_classifier_factory()
        await classifier_instance.initialize()

        result = await classifier_instance.classify_content(
            content="Unstructured text without clear indicators.", url=""
        )

        assert isinstance(result, classification_cls)
        assert (
            classifier_instance.embedding_manager.generate_embeddings.await_count == 1
        )
        assert result.confidence_scores
        assert all(0.0 <= score <= 1.0 for score in result.confidence_scores.values())

    @pytest.mark.asyncio
    async def test_semantic_classification_direct_call(
        self,
        classifier: ContentClassifier,
        content_type_enum: ContentType,
    ) -> None:
        """Call semantic classification helper directly for coverage.

        Args:
            classifier: Initialized classifier under test.
            content_type_enum: Enumeration representing content types.
        """
        scores = await classifier._semantic_classification(  # pylint: disable=protected-access
            "API reference documentation"
        )

        assert scores[content_type_enum.DOCUMENTATION] == pytest.approx(1.0)
        assert all(score >= 0.0 for score in scores.values())
