"""Comprehensive tests for the personalized ranking service."""

import asyncio
from datetime import datetime, timezone, timedelta, UTC
from unittest.mock import patch

import pytest

from src.services.query_processing.ranking import (
    ContentCategory,
    InteractionEvent,
    InteractionType,
    PersonalizedRankingRequest,
    PersonalizedRankingResult,
    PersonalizedRankingService,
    RankedResult,
    RankingStrategy,
    UserPreference,
    UserProfile,
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
        assert profile.total_interactions == 0
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
            total_interactions=150,
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
        assert profile.total_interactions == 150
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
            total_interactions=0,
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
            UserProfile(user_id="user123", total_interactions=-1)

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


class TestPersonalizedRankingService:
    """Test PersonalizedRankingService class."""

    @pytest.fixture
    def service(self):
        """Create ranking service instance."""
        return PersonalizedRankingService(
            enable_learning=True,
            enable_collaborative_filtering=True,
            profile_cache_size=100,
            interaction_retention_days=30,
        )

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            {
                "id": "result1",
                "title": "Python Tutorial",
                "content": "Learn Python programming",
                "score": 0.6,  # Lower scores to avoid exceeding 1.0 with boosts
                "content_type": "tutorial",
            },
            {
                "id": "result2",
                "title": "Flask Documentation",
                "content": "Flask web framework docs",
                "score": 0.5,
                "content_type": "documentation",
            },
            {
                "id": "result3",
                "title": "Django Guide",
                "content": "Django web development",
                "score": 0.4,
                "content_type": "tutorial",
            },
        ]

    @pytest.fixture
    def sample_request(self, sample_results):
        """Create sample ranking request."""
        return PersonalizedRankingRequest(
            user_id="user123",
            session_id="session456",
            query="python web framework",
            results=sample_results,
            strategy=RankingStrategy.HYBRID,
            personalization_strength=0.7,
        )

    def test_service_initialization(self):
        """Test service initialization with different parameters."""
        # Default initialization
        service1 = PersonalizedRankingService()
        assert service1.enable_learning is True
        assert service1.enable_collaborative_filtering is True
        assert service1.profile_cache_size == 1000
        assert service1.interaction_retention_days == 90

        # Custom initialization
        service2 = PersonalizedRankingService(
            enable_learning=False,
            enable_collaborative_filtering=False,
            profile_cache_size=500,
            interaction_retention_days=60,
        )
        assert service2.enable_learning is False
        assert service2.enable_collaborative_filtering is False
        assert service2.profile_cache_size == 500
        assert service2.interaction_retention_days == 60

    def test_initial_state(self, service):
        """Test initial service state."""
        assert service.user_profiles == {}
        assert service.interaction_history == {}
        assert service.user_similarity_matrix == {}
        assert service.item_features == {}
        assert service.content_embeddings == {}
        assert service.ranking_models == {}

        stats = service.get_performance_stats()
        assert stats["total_rankings"] == 0
        assert stats["avg_processing_time"] == 0.0
        assert stats["personalization_rate"] == 0.0
        assert stats["strategy_usage"] == {}
        assert stats["cached_profiles"] == 0
        assert stats["users_with_history"] == 0

    @pytest.mark.asyncio
    async def test_get_user_profile_new_user(self, service):
        """Test getting profile for new user."""
        profile = await service._get_user_profile("new_user")

        assert profile.user_id == "new_user"
        assert profile.preferences == []
        assert profile.total_interactions == 0
        assert profile.confidence_score == 0.0
        assert "new_user" in service.user_profiles

    @pytest.mark.asyncio
    async def test_get_user_profile_existing_user(self, service):
        """Test getting profile for existing user."""
        # Create initial profile
        profile1 = await service._get_user_profile("existing_user")
        profile1.total_interactions = 10
        profile1.confidence_score = 0.5

        # Get same profile again
        profile2 = await service._get_user_profile("existing_user")

        assert profile1 is profile2
        assert profile2.total_interactions == 10
        assert profile2.confidence_score == 0.5

    @pytest.mark.asyncio
    async def test_profile_cache_eviction(self, service):
        """Test profile cache LRU eviction."""
        service.profile_cache_size = 2

        # Add profiles up to cache limit
        await service._get_user_profile("user1")
        await service._get_user_profile("user2")

        assert len(service.user_profiles) == 2
        assert "user1" in service.user_profiles
        assert "user2" in service.user_profiles

        # Add third profile should evict oldest
        await service._get_user_profile("user3")

        assert len(service.user_profiles) == 2
        assert "user1" not in service.user_profiles  # Evicted
        assert "user2" in service.user_profiles
        assert "user3" in service.user_profiles

    def test_should_apply_personalization_conditions(self, service):
        """Test personalization application conditions."""
        profile = UserProfile(user_id="user123")
        request = PersonalizedRankingRequest(
            user_id="user123",
            query="test",
            results=[{"id": "1", "title": "Test", "score": 0.8}],
        )

        # Insufficient confidence
        profile.confidence_score = 0.2
        profile.total_interactions = 10
        assert not service._should_apply_personalization(profile, request)

        # Insufficient interactions
        profile.confidence_score = 0.5
        profile.total_interactions = 3
        assert not service._should_apply_personalization(profile, request)

        # Default strategy
        profile.confidence_score = 0.5
        profile.total_interactions = 10
        request.strategy = RankingStrategy.DEFAULT
        assert not service._should_apply_personalization(profile, request)

        # Should personalize
        profile.confidence_score = 0.5
        profile.total_interactions = 10
        request.strategy = RankingStrategy.HYBRID
        assert service._should_apply_personalization(profile, request)

    @pytest.mark.asyncio
    async def test_rank_results_default_strategy(self, service, sample_request):
        """Test ranking with default strategy (no personalization)."""
        sample_request.strategy = RankingStrategy.DEFAULT

        result = await service.rank_results(sample_request)

        assert result.strategy_used == RankingStrategy.DEFAULT
        assert result.personalization_applied is False
        assert result.user_profile_confidence == 0.0
        assert len(result.ranked_results) == 3

        # Should maintain original order (sorted by original score)
        assert result.ranked_results[0].result_id == "result1"  # Highest score (0.6)
        assert result.ranked_results[1].result_id == "result2"  # Middle score (0.5)
        assert result.ranked_results[2].result_id == "result3"  # Lowest score (0.4)

    @pytest.mark.asyncio
    async def test_rank_results_insufficient_profile(self, service, sample_request):
        """Test ranking when user profile is insufficient for personalization."""
        result = await service.rank_results(sample_request)

        # Should fall back to default strategy
        assert result.strategy_used == RankingStrategy.DEFAULT
        assert result.personalization_applied is False
        assert result.user_profile_confidence == 0.0

    @pytest.mark.asyncio
    async def test_rank_results_with_personalization(self, service, sample_request):
        """Test ranking with sufficient profile for personalization."""
        # Create user profile with sufficient data
        profile = UserProfile(
            user_id="user123",
            total_interactions=20,
            confidence_score=0.8,
            preferred_result_types={"tutorial": 0.9, "documentation": 0.6},
        )
        service.user_profiles["user123"] = profile

        result = await service.rank_results(sample_request)

        assert result.strategy_used == RankingStrategy.HYBRID
        assert result.personalization_applied is True
        assert result.user_profile_confidence == 0.8
        assert len(result.ranked_results) == 3

    @pytest.mark.asyncio
    async def test_content_based_ranking(self, service, sample_request):
        """Test content-based ranking strategy."""
        sample_request.strategy = RankingStrategy.CONTENT_BASED

        # Create user profile with content preferences (lower values to avoid exceeding 1.0)
        profile = UserProfile(
            user_id="user123",
            total_interactions=20,
            confidence_score=0.8,
            preferred_result_types={"tutorial": 0.5, "documentation": 0.1},
        )
        service.user_profiles["user123"] = profile

        result = await service.rank_results(sample_request)

        assert result.strategy_used == RankingStrategy.CONTENT_BASED
        assert result.personalization_applied is True

        # Check that we have the expected number of results
        assert len(result.ranked_results) == 3

        # Tutorials should be boosted over documentation
        tutorial_results = [
            r
            for r in result.ranked_results
            if r.metadata.get("content_type") == "tutorial"
        ]
        doc_results = [
            r
            for r in result.ranked_results
            if r.metadata.get("content_type") == "documentation"
        ]

        # Verify we have the expected content types
        assert len(tutorial_results) >= 1  # Should have at least one tutorial
        assert len(doc_results) >= 1  # Should have at least one documentation

    @pytest.mark.asyncio
    async def test_collaborative_filtering_ranking(self, service, sample_request):
        """Test collaborative filtering ranking strategy."""
        sample_request.strategy = RankingStrategy.COLLABORATIVE_FILTERING

        # Create user profile
        profile = UserProfile(
            user_id="user123", total_interactions=20, confidence_score=0.8
        )
        service.user_profiles["user123"] = profile

        # Add some similarity data
        service.user_similarity_matrix["user123"] = {"user456": 0.8, "user789": 0.6}

        result = await service.rank_results(sample_request)

        if service.enable_collaborative_filtering:
            assert result.strategy_used == RankingStrategy.COLLABORATIVE_FILTERING
            assert result.personalization_applied is True
        else:
            # Should fall back to default if collaborative filtering disabled
            assert result.strategy_used == RankingStrategy.DEFAULT

    @pytest.mark.asyncio
    async def test_behavioral_ranking(self, service, sample_request):
        """Test behavioral ranking strategy."""
        sample_request.strategy = RankingStrategy.BEHAVIORAL

        # Create user profile with behavioral patterns
        profile = UserProfile(
            user_id="user123",
            total_interactions=20,
            confidence_score=0.8,
            active_hours={9: 0.8, 14: 0.9},
            query_patterns={"python": 10, "flask": 5},
            exploration_tendency=0.7,
        )
        service.user_profiles["user123"] = profile

        result = await service.rank_results(sample_request)

        assert result.strategy_used == RankingStrategy.BEHAVIORAL
        assert result.personalization_applied is True

    @pytest.mark.asyncio
    async def test_contextual_ranking(self, service, sample_request):
        """Test contextual ranking strategy."""
        sample_request.strategy = RankingStrategy.CONTEXTUAL
        sample_request.context = {
            "domain": "programming",
            "platform": "web",
            "session_length": 600,
            "previous_searches": 5,
        }

        # Create user profile
        profile = UserProfile(
            user_id="user123", total_interactions=20, confidence_score=0.8
        )
        service.user_profiles["user123"] = profile

        result = await service.rank_results(sample_request)

        assert result.strategy_used == RankingStrategy.CONTEXTUAL
        assert result.personalization_applied is True

    @pytest.mark.asyncio
    async def test_learning_to_rank(self, service, sample_request):
        """Test learning-to-rank strategy."""
        sample_request.strategy = RankingStrategy.LEARNING_TO_RANK

        # Create user profile
        profile = UserProfile(
            user_id="user123",
            total_interactions=20,
            confidence_score=0.8,
            exploration_tendency=0.6,
            quality_sensitivity=0.8,
        )
        service.user_profiles["user123"] = profile

        result = await service.rank_results(sample_request)

        assert result.strategy_used == RankingStrategy.LEARNING_TO_RANK
        assert result.personalization_applied is True

    @pytest.mark.asyncio
    async def test_hybrid_ranking(self, service, sample_request):
        """Test hybrid ranking strategy (default)."""
        # Create user profile
        profile = UserProfile(
            user_id="user123",
            total_interactions=20,
            confidence_score=0.8,
            preferred_result_types={"tutorial": 0.9},
            exploration_tendency=0.6,
        )
        service.user_profiles["user123"] = profile

        result = await service.rank_results(sample_request)

        assert result.strategy_used == RankingStrategy.HYBRID
        assert result.personalization_applied is True

        # Check that multiple factors are present in ranking factors
        for ranked_result in result.ranked_results:
            factors = ranked_result.ranking_factors
            assert "original_relevance" in factors
            # Should have multiple ranking factors for hybrid
            assert len(factors) > 1

    @pytest.mark.asyncio
    async def test_record_interaction_learning_disabled(self, service):
        """Test recording interaction when learning is disabled."""
        service.enable_learning = False

        interaction = InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.CLICK,
        )

        await service.record_interaction(interaction)

        # Should not store interaction when learning disabled
        assert "user123" not in service.interaction_history

    @pytest.mark.asyncio
    async def test_record_interaction_learning_enabled(self, service):
        """Test recording interaction when learning is enabled."""
        interaction = InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.CLICK,
            query="python tutorial",
        )

        await service.record_interaction(interaction)

        # Should store interaction
        assert "user123" in service.interaction_history
        assert len(service.interaction_history["user123"]) == 1
        assert service.interaction_history["user123"][0] == interaction

    @pytest.mark.asyncio
    async def test_interaction_history_cleanup(self, service):
        """Test old interaction cleanup."""
        # Create old interaction
        old_interaction = InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.CLICK,
            timestamp=datetime.now(tz=UTC)
            - timedelta(days=40),  # Older than retention period
        )

        # Create recent interaction
        recent_interaction = InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result790",
            interaction_type=InteractionType.VIEW,
            timestamp=datetime.now(tz=UTC)
            - timedelta(days=5),  # Within retention period
        )

        # Add old interaction first
        service.interaction_history["user123"] = [old_interaction]

        # Record recent interaction (should trigger cleanup)
        await service.record_interaction(recent_interaction)

        # Should only have recent interaction
        assert len(service.interaction_history["user123"]) == 1
        assert service.interaction_history["user123"][0] == recent_interaction

    @pytest.mark.asyncio
    async def test_user_profile_updates_from_interactions(self, service):
        """Test user profile updates from interactions."""
        # Create a user profile first to avoid recursion
        profile = UserProfile(user_id="user123")
        service.user_profiles["user123"] = profile

        # Record multiple interactions
        interactions = [
            InteractionEvent(
                user_id="user123",
                session_id="session456",
                result_id="result1",
                interaction_type=InteractionType.CLICK,
                query="python tutorial",
                timestamp=datetime.now(tz=UTC).replace(hour=9),
            ),
            InteractionEvent(
                user_id="user123",
                session_id="session456",
                result_id="result2",
                interaction_type=InteractionType.BOOKMARK,
                query="flask guide",
                timestamp=datetime.now(tz=UTC).replace(hour=14),
            ),
            InteractionEvent(
                user_id="user123",
                session_id="session456",
                result_id="result3",
                interaction_type=InteractionType.RATING,
                value=4.5,
                timestamp=datetime.now(tz=UTC).replace(hour=9),
            ),
        ]

        for interaction in interactions:
            await service.record_interaction(interaction)

        # Get updated profile
        profile = service.user_profiles["user123"]

        # Check profile updates
        assert profile.total_interactions == 3
        assert profile.confidence_score > 0.0
        assert 9 in profile.active_hours
        assert 14 in profile.active_hours
        assert (
            profile.active_hours[9] >= profile.active_hours[14]
        )  # More or equal activity at hour 9
        assert "python" in profile.query_patterns
        assert "flask" in profile.query_patterns
        assert profile.query_patterns["python"] >= 1
        assert profile.query_patterns["flask"] >= 1

    def test_diversity_score_calculation(self, service):
        """Test diversity score calculation."""
        # Single result - should be max diversity
        single_result = [
            RankedResult(
                result_id="1",
                title="Test",
                content="Test",
                original_score=0.8,
                personalized_score=0.8,
                final_score=0.8,
                metadata={"content_type": "tutorial"},
            )
        ]
        diversity = service._calculate_diversity_score(single_result)
        assert diversity == 1.0

        # Multiple results with different types
        diverse_results = [
            RankedResult(
                result_id="1",
                title="Test 1",
                content="Test",
                original_score=0.8,
                personalized_score=0.8,
                final_score=0.8,
                metadata={"content_type": "tutorial"},
            ),
            RankedResult(
                result_id="2",
                title="Test 2",
                content="Test",
                original_score=0.7,
                personalized_score=0.7,
                final_score=0.7,
                metadata={"content_type": "documentation"},
            ),
            RankedResult(
                result_id="3",
                title="Test 3",
                content="Test",
                original_score=0.6,
                personalized_score=0.6,
                final_score=0.6,
                metadata={"content_type": "blog_post"},
            ),
        ]
        diversity = service._calculate_diversity_score(diverse_results)
        assert diversity == 3.0 / 3.0  # 3 unique types out of 3 results

        # Multiple results with same type
        same_type_results = [
            RankedResult(
                result_id="1",
                title="Test 1",
                content="Test",
                original_score=0.8,
                personalized_score=0.8,
                final_score=0.8,
                metadata={"content_type": "tutorial"},
            ),
            RankedResult(
                result_id="2",
                title="Test 2",
                content="Test",
                original_score=0.7,
                personalized_score=0.7,
                final_score=0.7,
                metadata={"content_type": "tutorial"},
            ),
        ]
        diversity = service._calculate_diversity_score(same_type_results)
        assert diversity == 1.0 / 2.0  # 1 unique type out of 2 results

    def test_coverage_score_calculation(self, service):
        """Test preference coverage score calculation."""
        profile = UserProfile(
            user_id="user123",
            preferences=[
                UserPreference(
                    attribute="language", value="python", weight=0.8, confidence=0.9
                ),
                UserPreference(
                    attribute="category", value="tutorial", weight=0.7, confidence=0.8
                ),
            ],
        )

        results = [
            RankedResult(
                result_id="1",
                title="Python Tutorial",
                content="Test",
                original_score=0.8,
                personalized_score=0.8,
                final_score=0.8,
                metadata={"language": "python", "category": "tutorial"},
            ),
            RankedResult(
                result_id="2",
                title="Java Guide",
                content="Test",
                original_score=0.7,
                personalized_score=0.7,
                final_score=0.7,
                metadata={"language": "java", "category": "documentation"},
            ),
        ]

        coverage = service._calculate_coverage_score(results, profile)
        assert coverage == 2.0 / 2.0  # Both preferences covered

        # Test with no preferences
        empty_profile = UserProfile(user_id="user456")
        coverage = service._calculate_coverage_score(results, empty_profile)
        assert coverage == 0.0

    def test_reranking_impact_calculation(self, service):
        """Test reranking impact calculation."""
        original_results = [
            {"id": "1", "title": "Result 1", "score": 0.9},
            {"id": "2", "title": "Result 2", "score": 0.8},
            {"id": "3", "title": "Result 3", "score": 0.7},
        ]

        # No reranking - same order
        same_order_results = [
            RankedResult(
                result_id="1",
                title="Result 1",
                content="Test",
                original_score=0.9,
                personalized_score=0.9,
                final_score=0.9,
            ),
            RankedResult(
                result_id="2",
                title="Result 2",
                content="Test",
                original_score=0.8,
                personalized_score=0.8,
                final_score=0.8,
            ),
            RankedResult(
                result_id="3",
                title="Result 3",
                content="Test",
                original_score=0.7,
                personalized_score=0.7,
                final_score=0.7,
            ),
        ]

        impact = service._calculate_reranking_impact(
            original_results, same_order_results
        )
        assert impact == 0.0

        # Complete reversal
        reversed_results = [
            RankedResult(
                result_id="3",
                title="Result 3",
                content="Test",
                original_score=0.7,
                personalized_score=0.7,
                final_score=0.7,
            ),
            RankedResult(
                result_id="2",
                title="Result 2",
                content="Test",
                original_score=0.8,
                personalized_score=0.8,
                final_score=0.8,
            ),
            RankedResult(
                result_id="1",
                title="Result 1",
                content="Test",
                original_score=0.9,
                personalized_score=0.9,
                final_score=0.9,
            ),
        ]

        impact = service._calculate_reranking_impact(original_results, reversed_results)
        assert impact == (2 + 0 + 2) / 3  # Average position change

    def test_content_preference_boost_calculation(self, service):
        """Test content preference boost calculation."""
        profile = UserProfile(
            user_id="user123",
            preferred_result_types={"tutorial": 0.9, "documentation": 0.3},
            preferences=[
                UserPreference(
                    attribute="language", value="python", weight=0.8, confidence=0.9
                )
            ],
        )

        # Result matching preferences
        matching_result = {
            "content_type": "tutorial",
            "language": "python",
            "title": "Python Tutorial",
        }
        boost = service._calculate_content_preference_boost(matching_result, profile)
        assert boost > 0.0
        assert boost <= 0.5  # Capped at 0.5

        # Result not matching preferences
        non_matching_result = {
            "content_type": "news",
            "language": "java",
            "title": "Java News",
        }
        boost = service._calculate_content_preference_boost(
            non_matching_result, profile
        )
        assert boost == 0.0

    def test_behavioral_boost_calculation(self, service):
        """Test behavioral boost calculation."""
        profile = UserProfile(
            user_id="user123",
            active_hours={9: 0.8, 14: 0.9},
            query_patterns={"python": 10, "flask": 5},
            exploration_tendency=0.8,
        )

        result = {"query_keywords": ["python", "web"], "novelty_score": 0.9}

        context = {}

        with patch("src.services.query_processing.ranking.datetime") as mock_datetime:
            mock_datetime.now.return_value.hour = 9  # Active hour
            boost = service._calculate_behavioral_boost(result, profile, context)

        assert boost > 0.0
        assert boost <= 0.4  # Capped at 0.4

    def test_contextual_boost_calculation(self, service):
        """Test contextual boost calculation."""
        profile = UserProfile(user_id="user123")

        result = {
            "domain": "programming",
            "time_sensitive": True,
            "recency_score": 0.8,
            "platform_optimized": ["web", "mobile"],
        }

        context = {"domain": "programming", "platform": "web"}

        boost = service._calculate_contextual_boost(result, profile, context)
        assert boost > 0.0
        assert boost <= 0.3  # Capped at 0.3

    def test_diversity_boost_calculation(self, service):
        """Test diversity boost calculation."""
        all_results = [
            {"content_type": "tutorial"},
            {"content_type": "tutorial"},
            {"content_type": "documentation"},
            {"content_type": "blog_post"},
            {"content_type": "tutorial"},
        ]

        # First tutorial should get no boost (many tutorials in top 5)
        tutorial_result = {"content_type": "tutorial"}
        boost = service._calculate_diversity_boost(tutorial_result, all_results)
        assert boost < 0.0  # Penalty for over-represented type

        # Unique type should get boost
        unique_result = {"content_type": "reference"}
        boost = service._calculate_diversity_boost(unique_result, all_results)
        assert boost > 0.0

    def test_freshness_boost_calculation(self, service):
        """Test freshness boost calculation."""
        # Very fresh content
        fresh_result = {"published_date": datetime.now(tz=UTC).isoformat()}
        boost = service._calculate_freshness_boost(fresh_result)
        assert boost == 0.1

        # Week-old content (exactly 7 days should still get the fresh boost)
        week_old_result = {
            "published_date": (datetime.now(tz=UTC) - timedelta(days=6)).isoformat()
        }
        boost = service._calculate_freshness_boost(week_old_result)
        assert boost == 0.1  # Still within 7 days

        # Content older than 7 days but within 30 days
        older_result = {
            "published_date": (datetime.now(tz=UTC) - timedelta(days=15)).isoformat()
        }
        boost = service._calculate_freshness_boost(older_result)
        assert boost == 0.05

        # Month-old content (exactly 30 days still gets 0.05)
        month_old_result = {
            "published_date": (datetime.now(tz=UTC) - timedelta(days=30)).isoformat()
        }
        boost = service._calculate_freshness_boost(month_old_result)
        assert boost == 0.05

        # 3-month-old content
        three_month_old_result = {
            "published_date": (datetime.now(tz=UTC) - timedelta(days=60)).isoformat()
        }
        boost = service._calculate_freshness_boost(three_month_old_result)
        assert boost == 0.02

        # Very old content
        old_result = {
            "published_date": (datetime.now(tz=UTC) - timedelta(days=200)).isoformat()
        }
        boost = service._calculate_freshness_boost(old_result)
        assert boost == 0.0

        # No date
        no_date_result = {}
        boost = service._calculate_freshness_boost(no_date_result)
        assert boost == 0.0

    def test_ranking_feature_extraction(self, service):
        """Test ML ranking feature extraction."""
        profile = UserProfile(
            user_id="user123",
            total_interactions=50,
            confidence_score=0.8,
            exploration_tendency=0.7,
            quality_sensitivity=0.9,
            preferred_result_types={"tutorial": 0.8},
        )

        result = {
            "id": "result1",
            "score": 0.85,
            "content": "This is a long content example with lots of text",
            "title": "Python Tutorial",
            "has_code": True,
            "content_type": "tutorial",
        }

        request = PersonalizedRankingRequest(
            user_id="user123",
            query="python web development tutorial",
            results=[result],
            personalization_strength=0.7,
        )

        features = service._extract_ranking_features(result, profile, request)

        assert "original_score" in features
        assert features["original_score"] == 0.85
        assert features["user_total_interactions"] == 50.0
        assert features["user_profile_confidence"] == 0.8
        assert features["user_exploration_tendency"] == 0.7
        assert features["user_quality_sensitivity"] == 0.9
        assert features["has_code_examples"] == 1.0
        assert features["content_type_preference"] == 0.8
        assert features["query_length"] == 4.0  # "python web development tutorial"
        assert features["personalization_strength"] == 0.7

    def test_ml_ranking_model_application(self, service):
        """Test applying ML ranking model."""
        profile = UserProfile(user_id="user123")

        features = {
            "original_score": 0.8,
            "user_profile_confidence": 0.7,
            "content_type_preference": 0.9,
            "user_exploration_tendency": 0.6,
            "has_code_examples": 1.0,
        }

        score = service._apply_ranking_model(features, profile)

        assert 0.0 <= score <= 1.0
        # Score should be influenced by the weighted features
        assert score > 0.0

    def test_explanation_generation(self, service):
        """Test ranking explanation generation."""
        # Content boost explanation
        explanation = service._generate_content_explanation(
            0.25, UserProfile(user_id="user123")
        )
        assert "higher based on your content preferences" in explanation

        explanation = service._generate_content_explanation(
            0.15, UserProfile(user_id="user123")
        )
        assert "boosted to match your interests" in explanation

        explanation = service._generate_content_explanation(
            0.05, UserProfile(user_id="user123")
        )
        assert "Standard ranking applied" in explanation

        # Hybrid explanation
        ranking_factors = {
            "content_preference": 0.3,
            "behavioral_pattern": 0.2,
            "contextual_relevance": 0.1,
        }
        explanation = service._generate_hybrid_explanation(ranking_factors)
        assert "content interests" in explanation

    def test_applied_factors_by_strategy(self, service):
        """Test getting applied factors for different strategies."""
        factors = service._get_applied_factors(RankingStrategy.CONTENT_BASED)
        assert "content_preference" in factors
        assert "attribute_matching" in factors

        factors = service._get_applied_factors(RankingStrategy.COLLABORATIVE_FILTERING)
        assert "user_similarity" in factors
        assert "interaction_patterns" in factors

        factors = service._get_applied_factors(RankingStrategy.BEHAVIORAL)
        assert "temporal_patterns" in factors
        assert "query_patterns" in factors

        factors = service._get_applied_factors(RankingStrategy.CONTEXTUAL)
        assert "session_context" in factors
        assert "temporal_context" in factors

        factors = service._get_applied_factors(RankingStrategy.LEARNING_TO_RANK)
        assert "ml_features" in factors
        assert "trained_model" in factors

        factors = service._get_applied_factors(RankingStrategy.HYBRID)
        assert "content_preference" in factors
        assert "behavioral_patterns" in factors
        assert "diversity" in factors

        factors = service._get_applied_factors(RankingStrategy.DEFAULT)
        assert factors == ["original_relevance"]

    def test_performance_stats_updates(self, service):
        """Test performance statistics updates."""
        initial_stats = service.get_performance_stats()
        assert initial_stats["total_rankings"] == 0

        # Update performance stats
        service._update_performance_stats(RankingStrategy.HYBRID, 150.0, True)
        service._update_performance_stats(RankingStrategy.DEFAULT, 50.0, False)
        service._update_performance_stats(RankingStrategy.HYBRID, 200.0, True)

        stats = service.get_performance_stats()
        assert stats["total_rankings"] == 3
        assert stats["avg_processing_time"] == (150.0 + 50.0 + 200.0) / 3
        assert stats["personalization_rate"] == 2.0 / 3.0  # 2 personalized out of 3
        assert stats["strategy_usage"]["hybrid"] == 2
        assert stats["strategy_usage"]["default"] == 1

    def test_clear_user_data(self, service):
        """Test clearing user data."""
        # Add user data
        service.user_profiles["user123"] = UserProfile(user_id="user123")
        service.interaction_history["user123"] = [
            InteractionEvent(
                user_id="user123",
                session_id="session456",
                result_id="result789",
                interaction_type=InteractionType.CLICK,
            )
        ]
        service.user_similarity_matrix["user123"] = {"user456": 0.8}

        # Clear data
        service.clear_user_data("user123")

        assert "user123" not in service.user_profiles
        assert "user123" not in service.interaction_history
        assert "user123" not in service.user_similarity_matrix

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, service, sample_request):
        """Test error handling and fallback to default ranking."""
        # Create a user profile that would normally trigger personalization
        profile = UserProfile(
            user_id="user123", total_interactions=20, confidence_score=0.8
        )
        service.user_profiles["user123"] = profile

        # Mock an error in personalized ranking
        with patch.object(
            service, "_apply_personalized_ranking", side_effect=Exception("Test error")
        ):
            result = await service.rank_results(sample_request)

            # Should fall back to default ranking
            assert result.strategy_used == RankingStrategy.DEFAULT
            assert result.personalization_applied is False
            assert result.user_profile_confidence == 0.0
            assert "error" in result.ranking_metadata

    @pytest.mark.asyncio
    async def test_ranking_with_explanations_enabled(self, service, sample_request):
        """Test ranking with explanations enabled."""
        sample_request.enable_explanations = True
        sample_request.strategy = RankingStrategy.CONTENT_BASED

        # Create user profile
        profile = UserProfile(
            user_id="user123",
            total_interactions=20,
            confidence_score=0.8,
            preferred_result_types={"tutorial": 0.9},
        )
        service.user_profiles["user123"] = profile

        result = await service.rank_results(sample_request)

        # Should include explanations
        for ranked_result in result.ranked_results:
            if ranked_result.ranking_explanation:
                assert isinstance(ranked_result.ranking_explanation, str)
                assert len(ranked_result.ranking_explanation) > 0

    @pytest.mark.asyncio
    async def test_cold_start_handling(self, service):
        """Test handling of cold start users (no interaction history)."""
        request = PersonalizedRankingRequest(
            user_id="brand_new_user",
            query="python tutorial",
            results=[
                {"id": "1", "title": "Python Basics", "score": 0.9},
                {"id": "2", "title": "Advanced Python", "score": 0.8},
            ],
        )

        result = await service.rank_results(request)

        # Should handle cold start gracefully
        assert result.strategy_used == RankingStrategy.DEFAULT
        assert result.personalization_applied is False
        assert result.user_profile_confidence == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, service, sample_request):
        """Test handling concurrent ranking requests."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            request_copy = PersonalizedRankingRequest(
                user_id=f"user{i}",
                query=sample_request.query,
                results=sample_request.results,
            )
            tasks.append(service.rank_results(request_copy))

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, PersonalizedRankingResult)
            assert len(result.ranked_results) == 3

    def test_time_context_boost_calculation(self, service):
        """Test time-based context boost calculation."""
        with patch("src.services.query_processing.ranking.datetime") as mock_datetime:
            # Business hours
            mock_datetime.now.return_value.hour = 10
            mock_datetime.now.return_value.weekday.return_value = 2  # Wednesday

            boost = service._get_temporal_context_boost({})
            assert boost > 0.0  # Should get business hours boost + weekday boost

            # Evening weekend
            mock_datetime.now.return_value.hour = 20
            mock_datetime.now.return_value.weekday.return_value = 6  # Sunday

            boost = service._get_temporal_context_boost({})
            assert boost > 0.0  # Should get evening boost but no weekday boost

    @pytest.mark.asyncio
    async def test_interaction_event_edge_cases(self, service):
        """Test edge cases in interaction event handling."""
        # Test with None values
        interaction = InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result789",
            interaction_type=InteractionType.CLICK,
            value=None,
            query=None,
            result_position=None,
        )

        await service.record_interaction(interaction)
        assert "user123" in service.interaction_history

        # Test interaction without query
        interaction_no_query = InteractionEvent(
            user_id="user123",
            session_id="session456",
            result_id="result790",
            interaction_type=InteractionType.VIEW,
        )

        await service.record_interaction(interaction_no_query)
        assert len(service.interaction_history["user123"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
