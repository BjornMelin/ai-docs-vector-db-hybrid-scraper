"""Enum tests for the personalized ranking service.

Tests for ranking strategy enums, interaction types, content categories and user prefs.
"""

from datetime import UTC, datetime

import pytest

from src.services.query_processing.ranking import (
    ContentCategory,
    InteractionType,
    RankingStrategy,
    UserPreference,
)


class TestRankingStrategy:
    """Test RankingStrategy enum."""

    def test_all_strategies_defined(self):
        """Test that all ranking strategies are defined."""
        expected_strategies = [
            "collaborative_filtering",
            "content_based",
            "hybrid",
            "learning_to_rank",
            "behavioral",
            "contextual",
            "default",
        ]

        for strategy in expected_strategies:
            assert hasattr(RankingStrategy, strategy.upper())
            assert getattr(RankingStrategy, strategy.upper()).value == strategy

    def test_strategy_values(self):
        """Test specific strategy values."""
        assert (
            RankingStrategy.COLLABORATIVE_FILTERING.value == "collaborative_filtering"
        )
        assert RankingStrategy.CONTENT_BASED.value == "content_based"
        assert RankingStrategy.HYBRID.value == "hybrid"
        assert RankingStrategy.LEARNING_TO_RANK.value == "learning_to_rank"
        assert RankingStrategy.BEHAVIORAL.value == "behavioral"
        assert RankingStrategy.CONTEXTUAL.value == "contextual"
        assert RankingStrategy.DEFAULT.value == "default"


class TestInteractionType:
    """Test InteractionType enum."""

    def test_all_interaction_types_defined(self):
        """Test that all interaction types are defined."""
        expected_types = [
            "click",
            "view",
            "download",
            "bookmark",
            "share",
            "dwell_time",
            "rating",
            "skip",
            "negative_feedback",
        ]

        for interaction_type in expected_types:
            assert hasattr(InteractionType, interaction_type.upper())
            assert (
                getattr(InteractionType, interaction_type.upper()).value
                == interaction_type
            )

    def test_interaction_type_values(self):
        """Test specific interaction type values."""
        assert InteractionType.CLICK.value == "click"
        assert InteractionType.VIEW.value == "view"
        assert InteractionType.DOWNLOAD.value == "download"
        assert InteractionType.BOOKMARK.value == "bookmark"
        assert InteractionType.SHARE.value == "share"
        assert InteractionType.DWELL_TIME.value == "dwell_time"
        assert InteractionType.RATING.value == "rating"
        assert InteractionType.SKIP.value == "skip"
        assert InteractionType.NEGATIVE_FEEDBACK.value == "negative_feedback"


class TestContentCategory:
    """Test ContentCategory enum."""

    def test_all_categories_defined(self):
        """Test that all content categories are defined."""
        expected_categories = [
            "programming",
            "documentation",
            "tutorial",
            "blog_post",
            "academic",
            "news",
            "reference",
            "troubleshooting",
            "best_practices",
            "examples",
            "tools",
            "frameworks",
        ]

        for category in expected_categories:
            assert hasattr(ContentCategory, category.upper())
            assert getattr(ContentCategory, category.upper()).value == category

    def test_category_values(self):
        """Test specific category values."""
        assert ContentCategory.PROGRAMMING.value == "programming"
        assert ContentCategory.DOCUMENTATION.value == "documentation"
        assert ContentCategory.TUTORIAL.value == "tutorial"
        assert ContentCategory.BLOG_POST.value == "blog_post"
        assert ContentCategory.ACADEMIC.value == "academic"


class TestUserPreference:
    """Test UserPreference model."""

    def test_valid_preference_creation(self):
        """Test creating valid user preference."""
        preference = UserPreference(
            attribute="content_type",
            value="tutorial",
            weight=0.8,
            confidence=0.9,
            learned_from=["click", "bookmark"],
            last_updated=datetime.now(tz=UTC),
        )

        assert preference.attribute == "content_type"
        assert preference.value == "tutorial"
        assert preference.weight == 0.8
        assert preference.confidence == 0.9
        assert preference.learned_from == ["click", "bookmark"]
        assert isinstance(preference.last_updated, datetime)

    def test_default_values(self):
        """Test default values for optional fields."""
        preference = UserPreference(
            attribute="language", value="python", weight=0.7, confidence=0.8
        )

        assert preference.learned_from == []
        assert isinstance(preference.last_updated, datetime)

    def test_weight_validation(self):
        """Test weight validation constraints."""
        # Valid weights
        UserPreference(attribute="test", value="test", weight=0.0, confidence=0.5)
        UserPreference(attribute="test", value="test", weight=1.0, confidence=0.5)
        UserPreference(attribute="test", value="test", weight=0.5, confidence=0.5)

        # Invalid weights
        with pytest.raises(ValueError):
            UserPreference(attribute="test", value="test", weight=-0.1, confidence=0.5)

        with pytest.raises(ValueError):
            UserPreference(attribute="test", value="test", weight=1.1, confidence=0.5)

    def test_confidence_validation(self):
        """Test confidence validation constraints."""
        # Valid confidence
        UserPreference(attribute="test", value="test", weight=0.5, confidence=0.0)
        UserPreference(attribute="test", value="test", weight=0.5, confidence=1.0)
        UserPreference(attribute="test", value="test", weight=0.5, confidence=0.5)

        # Invalid confidence
        with pytest.raises(ValueError):
            UserPreference(attribute="test", value="test", weight=0.5, confidence=-0.1)

        with pytest.raises(ValueError):
            UserPreference(attribute="test", value="test", weight=0.5, confidence=1.1)

    def test_numeric_and_string_values(self):
        """Test both numeric and string values."""
        # String value
        pref1 = UserPreference(
            attribute="category", value="programming", weight=0.8, confidence=0.9
        )
        assert pref1.value == "programming"

        # Numeric value
        pref2 = UserPreference(
            attribute="difficulty", value=3.5, weight=0.7, confidence=0.8
        )
        assert pref2.value == 3.5
