"""Comprehensive tests for the personalized ranking service."""

import pytest

from src.services.query_processing.ranking import (
    InteractionType,
    RankingStrategy,
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


class TestInteractionEvent:
    """Test InteractionEvent model."""

    def test_valid_interaction_creation(self):
        """Test creating valid interaction event."""
        interaction = InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.CLICK,
            timestamp=datetime.now(tz=UTC),
            value=None,
            query="python tutorial",
            result_position=1,
            context={"source": "search"},
        )

        assert interaction.user_id == "user123"
        assert interaction.session_id == "session456"
        assert interaction.result_id == "result789"
        assert interaction.interaction_type == InteractionType.CLICK
        assert isinstance(interaction.timestamp, datetime)
        assert interaction.value is None
        assert interaction.query == "python tutorial"
        assert interaction.result_position == 1
        assert interaction.context == {"source": "search"}

    def test_default_values(self):
        """Test default values for optional fields."""
        interaction = InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.VIEW,
        )

        assert isinstance(interaction.timestamp, datetime)
        assert interaction.value is None
        assert interaction.query is None
        assert interaction.result_position is None
        assert interaction.context == {}

    def test_rating_validation(self):
        """Test rating value validation."""
        # Valid ratings
        InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.RATING,
            value=1.0,
        )

        InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.RATING,
            value=5.0,
        )

        # Invalid ratings
        with pytest.raises(ValueError, match="Rating must be between 1.0 and 5.0"):
            InteractionEvent(
                user_id="user123",
                session_id="session456",
                result_id="result789",
                interaction_type=InteractionType.RATING,
                value=0.5,
            )

        with pytest.raises(ValueError, match="Rating must be between 1.0 and 5.0"):
            InteractionEvent(
                user_id="user123",
                session_id="session456",
                result_id="result789",
                interaction_type=InteractionType.RATING,
                value=5.5,
            )

    def test_dwell_time_validation(self):
        """Test dwell time validation."""
        # Valid dwell time
        InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.DWELL_TIME,
            value=120.5,
        )

        # Zero dwell time
        InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.DWELL_TIME,
            value=0.0,
        )

        # Invalid dwell time
        with pytest.raises(ValueError, match="Dwell time cannot be negative"):
            InteractionEvent(
                user_id="user123",
                session_id="session456",
                result_id="result789",
                interaction_type=InteractionType.DWELL_TIME,
                value=-1.0,
            )

    def test_position_validation(self):
        """Test result position validation."""
        # Valid positions
        InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.CLICK,
            result_position=0,
        )

        InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.CLICK,
            result_position=10,
        )

        # Invalid position
        with pytest.raises(ValueError):
            InteractionEvent(
                user_id="user123",
                session_id="session456",
                result_id="result789",
                interaction_type=InteractionType.CLICK,
                result_position=-1,
            )

    def test_other_interaction_types_no_validation(self):
        """Test that other interaction types don't validate value."""
        # Should not raise validation errors
        InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.CLICK,
            value=999,  # No validation for clicks
        )

        InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.BOOKMARK,
            value=-100,  # No validation for bookmarks
        )


class TestUserProfile:
    """Test UserProfile model."""

    def test_minimal_profile_creation(self):
        """Test creating profile with minimal required fields."""
        profile = UserProfile(user_id="user123")

        assert profile.user_id == "user123"
        assert profile.preferences == []
        assert profile.avg_session_length == 0.0
        assert profile.preferred_result_types == {}
        assert profile.query_patterns == {}
        assert profile.active_hours == {}
        assert profile.interaction_velocity == 0.0
        assert profile.exploration_tendency == 0.5
        assert profile.quality_sensitivity == 0.5
        assert profile._total_interactions == 0
        assert isinstance(profile.profile_created, datetime)
        assert isinstance(profile.last_updated, datetime)
        assert profile.confidence_score == 0.0

    def test_complete_profile_creation(self):
        """Test creating profile with all fields."""
        preferences = [
            UserPreference(
                attribute="language", value="python", weight=0.8, confidence=0.9
            )
        ]

        profile = UserProfile(
            user_id="user123",
            preferences=preferences,
            avg_session_length=300.5,
            preferred_result_types={"tutorial": 0.8, "documentation": 0.6},
            query_patterns={"python": 10, "flask": 5},
            active_hours={9: 0.8, 14: 0.9, 20: 0.7},
            interaction_velocity=2.5,
            exploration_tendency=0.7,
            quality_sensitivity=0.8,
            _total_interactions=150,
            confidence_score=0.9,
        )

        assert profile.user_id == "user123"
        assert len(profile.preferences) == 1
        assert profile.preferences[0].attribute == "language"
        assert profile.avg_session_length == 300.5
        assert profile.preferred_result_types == {"tutorial": 0.8, "documentation": 0.6}
        assert profile.query_patterns == {"python": 10, "flask": 5}
        assert profile.active_hours == {9: 0.8, 14: 0.9, 20: 0.7}
        assert profile.interaction_velocity == 2.5
        assert profile.exploration_tendency == 0.7
        assert profile.quality_sensitivity == 0.8
        assert profile._total_interactions == 150
        assert profile.confidence_score == 0.9

    def test_profile_validation_constraints(self):
        """Test profile field validation constraints."""
        # Valid ranges
        profile = UserProfile(
            user_id="user123",
            avg_session_length=0.0,
            interaction_velocity=0.0,
            exploration_tendency=0.0,
            quality_sensitivity=1.0,
            _total_interactions=0,
            confidence_score=1.0,
        )
        assert profile.user_id == "user123"

        # Invalid ranges
        with pytest.raises(ValueError):
            UserProfile(user_id="user123", avg_session_length=-1.0)

        with pytest.raises(ValueError):
            UserProfile(user_id="user123", interaction_velocity=-1.0)

        with pytest.raises(ValueError):
            UserProfile(user_id="user123", exploration_tendency=-0.1)

        with pytest.raises(ValueError):
            UserProfile(user_id="user123", exploration_tendency=1.1)

        with pytest.raises(ValueError):
            UserProfile(user_id="user123", quality_sensitivity=-0.1)

        with pytest.raises(ValueError):
            UserProfile(user_id="user123", quality_sensitivity=1.1)

        with pytest.raises(ValueError):
            UserProfile(user_id="user123", _total_interactions=-1)

        with pytest.raises(ValueError):
            UserProfile(user_id="user123", confidence_score=-0.1)

        with pytest.raises(ValueError):
            UserProfile(user_id="user123", confidence_score=1.1)


class TestRankedResult:
    """Test RankedResult model."""

    def test_minimal_result_creation(self):
        """Test creating result with minimal required fields."""
        result = RankedResult(
            result_id="result123",
            title="Test Result",
            content="Test content",
            original_score=0.8,
            personalized_score=0.85,
            final_score=0.9,
        )

        assert result.result_id == "result123"
        assert result.title == "Test Result"
        assert result.content == "Test content"
        assert result.original_score == 0.8
        assert result.personalized_score == 0.85
        assert result.final_score == 0.9
        assert result.ranking_factors == {}
        assert result.personalization_boost == 0.0
        assert result.metadata == {}
        assert result.ranking_explanation is None

    def test_complete_result_creation(self):
        """Test creating result with all fields."""
        result = RankedResult(
            result_id="result123",
            title="Test Result",
            content="Test content",
            original_score=0.8,
            personalized_score=0.85,
            final_score=0.9,
            ranking_factors={"content_preference": 0.1, "behavioral": 0.05},
            personalization_boost=0.1,
            metadata={"source": "test", "category": "tutorial"},
            ranking_explanation="Boosted due to user preferences",
        )

        assert result.ranking_factors == {"content_preference": 0.1, "behavioral": 0.05}
        assert result.personalization_boost == 0.1
        assert result.metadata == {"source": "test", "category": "tutorial"}
        assert result.ranking_explanation == "Boosted due to user preferences"

    def test_score_validation(self):
        """Test score field validation."""
        # Valid scores
        RankedResult(
            result_id="result123",
            title="Test",
            content="Test",
            original_score=0.0,
            personalized_score=0.0,
            final_score=0.0,
        )

        RankedResult(
            result_id="result123",
            title="Test",
            content="Test",
            original_score=1.0,
            personalized_score=1.0,
            final_score=1.0,
        )

        # Invalid scores
        with pytest.raises(ValueError):
            RankedResult(
                result_id="result123",
                title="Test",
                content="Test",
                original_score=-0.1,
                personalized_score=0.5,
                final_score=0.5,
            )

        with pytest.raises(ValueError):
            RankedResult(
                result_id="result123",
                title="Test",
                content="Test",
                original_score=1.1,
                personalized_score=0.5,
                final_score=0.5,
            )

    def test_personalization_boost_validation(self):
        """Test personalization boost validation."""
        # Valid boost range
        RankedResult(
            result_id="result123",
            title="Test",
            content="Test",
            original_score=0.5,
            personalized_score=0.5,
            final_score=0.5,
            personalization_boost=-1.0,
        )

        RankedResult(
            result_id="result123",
            title="Test",
            content="Test",
            original_score=0.5,
            personalized_score=0.5,
            final_score=0.5,
            personalization_boost=1.0,
        )

        # Invalid boost range
        with pytest.raises(ValueError):
            RankedResult(
                result_id="result123",
                title="Test",
                content="Test",
                original_score=0.5,
                personalized_score=0.5,
                final_score=0.5,
                personalization_boost=-1.1,
            )

        with pytest.raises(ValueError):
            RankedResult(
                result_id="result123",
                title="Test",
                content="Test",
                original_score=0.5,
                personalized_score=0.5,
                final_score=0.5,
                personalization_boost=1.1,
            )


class TestPersonalizedRankingRequest:
    """Test PersonalizedRankingRequest model."""

    def test_minimal_request_creation(self):
        """Test creating request with minimal required fields."""
        results = [
            {"id": "1", "title": "Result 1", "score": 0.8},
            {"id": "2", "title": "Result 2", "score": 0.7},
        ]

        request = PersonalizedRankingRequest(
            user_id="user123", query="python tutorial", results=results
        )

        assert request.user_id == "user123"
        assert request.session_id is None
        assert request.query == "python tutorial"
        assert len(request.results) == 2
        assert request.strategy == RankingStrategy.HYBRID
        assert request.personalization_strength == 0.7
        assert request.context == {}
        assert request.min_confidence_threshold == 0.3
        assert request.diversity_factor == 0.1
        assert request.freshness_factor == 0.1
        assert request.enable_explanations is False
        assert request.enable_ab_testing is False
        assert request.max_processing_time_ms == 1000.0

    def test_complete_request_creation(self):
        """Test creating request with all fields."""
        results = [
            {"id": "1", "title": "Result 1", "score": 0.8, "content": "Content 1"},
            {"id": "2", "title": "Result 2", "score": 0.7, "content": "Content 2"},
        ]

        request = PersonalizedRankingRequest(
            user_id="user123",
            session_id="session456",
            query="python tutorial",
            results=results,
            strategy=RankingStrategy.CONTENT_BASED,
            personalization_strength=0.8,
            context={"domain": "programming", "platform": "web"},
            min_confidence_threshold=0.5,
            diversity_factor=0.2,
            freshness_factor=0.15,
            enable_explanations=True,
            enable_ab_testing=True,
            max_processing_time_ms=2000.0,
        )

        assert request.user_id == "user123"
        assert request.session_id == "session456"
        assert request.query == "python tutorial"
        assert len(request.results) == 2
        assert request.strategy == RankingStrategy.CONTENT_BASED
        assert request.personalization_strength == 0.8
        assert request.context == {"domain": "programming", "platform": "web"}
        assert request.min_confidence_threshold == 0.5
        assert request.diversity_factor == 0.2
        assert request.freshness_factor == 0.15
        assert request.enable_explanations is True
        assert request.enable_ab_testing is True
        assert request.max_processing_time_ms == 2000.0

    def test_results_validation(self):
        """Test results validation."""
        # Empty results should fail
        with pytest.raises(ValueError, match="Results list cannot be empty"):
            PersonalizedRankingRequest(user_id="user123", query="test", results=[])

        # Missing required fields should fail
        with pytest.raises(ValueError, match="Results must contain fields"):
            PersonalizedRankingRequest(
                user_id="user123",
                query="test",
                results=[{"id": "1", "title": "Test"}],  # Missing score
            )

        with pytest.raises(ValueError, match="Results must contain fields"):
            PersonalizedRankingRequest(
                user_id="user123",
                query="test",
                results=[{"id": "1", "score": 0.8}],  # Missing title
            )

        with pytest.raises(ValueError, match="Results must contain fields"):
            PersonalizedRankingRequest(
                user_id="user123",
                query="test",
                results=[{"title": "Test", "score": 0.8}],  # Missing id
            )

    def test_field_validation_ranges(self):
        """Test field validation ranges."""
        results = [{"id": "1", "title": "Test", "score": 0.8}]

        # Valid ranges
        PersonalizedRankingRequest(
            user_id="user123",
            query="test",
            results=results,
            personalization_strength=0.0,
            min_confidence_threshold=0.0,
            diversity_factor=0.0,
            freshness_factor=0.0,
            max_processing_time_ms=100.0,
        )

        PersonalizedRankingRequest(
            user_id="user123",
            query="test",
            results=results,
            personalization_strength=1.0,
            min_confidence_threshold=1.0,
            diversity_factor=1.0,
            freshness_factor=1.0,
            max_processing_time_ms=5000.0,
        )

        # Invalid ranges
        with pytest.raises(ValueError):
            PersonalizedRankingRequest(
                user_id="user123",
                query="test",
                results=results,
                personalization_strength=-0.1,
            )

        with pytest.raises(ValueError):
            PersonalizedRankingRequest(
                user_id="user123",
                query="test",
                results=results,
                max_processing_time_ms=50.0,  # Below minimum
            )


class TestPersonalizedRankingResult:
    """Test PersonalizedRankingResult model."""

    def test_complete_result_creation(self):
        """Test creating complete ranking result."""
        ranked_results = [
            RankedResult(
                result_id="1",
                title="Result 1",
                content="Content 1",
                original_score=0.8,
                personalized_score=0.85,
                final_score=0.9,
            )
        ]

        result = PersonalizedRankingResult(
            ranked_results=ranked_results,
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            user_profile_confidence=0.8,
            processing_time_ms=150.5,
            reranking_impact=1.2,
            diversity_score=0.7,
            coverage_score=0.85,
            ranking_metadata={"factors_applied": ["content", "behavioral"]},
        )

        assert len(result.ranked_results) == 1
        assert result.strategy_used == RankingStrategy.HYBRID
        assert result.personalization_applied is True
        assert result.user_profile_confidence == 0.8
        assert result.processing_time_ms == 150.5
        assert result.reranking_impact == 1.2
        assert result.diversity_score == 0.7
        assert result.coverage_score == 0.85
        assert result.ranking_metadata == {"factors_applied": ["content", "behavioral"]}

    def test_field_validation_ranges(self):
        """Test field validation ranges."""
        ranked_results = [
            RankedResult(
                result_id="1",
                title="Result 1",
                content="Content 1",
                original_score=0.8,
                personalized_score=0.85,
                final_score=0.9,
            )
        ]

        # Valid ranges
        PersonalizedRankingResult(
            ranked_results=ranked_results,
            strategy_used=RankingStrategy.DEFAULT,
            personalization_applied=False,
            user_profile_confidence=0.0,
            processing_time_ms=0.0,
            reranking_impact=0.0,
            diversity_score=0.0,
            coverage_score=0.0,
        )

        PersonalizedRankingResult(
            ranked_results=ranked_results,
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            user_profile_confidence=1.0,
            processing_time_ms=10000.0,
            reranking_impact=100.0,
            diversity_score=1.0,
            coverage_score=1.0,
        )

        # Invalid ranges
        with pytest.raises(ValueError):
            PersonalizedRankingResult(
                ranked_results=ranked_results,
                strategy_used=RankingStrategy.HYBRID,
                personalization_applied=True,
                user_profile_confidence=-0.1,
                processing_time_ms=100.0,
                reranking_impact=1.0,
                diversity_score=0.5,
                coverage_score=0.5,
            )

        with pytest.raises(ValueError):
            PersonalizedRankingResult(
                ranked_results=ranked_results,
                strategy_used=RankingStrategy.HYBRID,
                personalization_applied=True,
                user_profile_confidence=0.8,
                processing_time_ms=-1.0,
                reranking_impact=1.0,
                diversity_score=0.5,
                coverage_score=0.5,
            )


# Import split test classes
from .test_ranking_service_calculations import (
    TestPersonalizedRankingServiceCalculations,
)
from .test_ranking_service_core import TestPersonalizedRankingServiceCore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
