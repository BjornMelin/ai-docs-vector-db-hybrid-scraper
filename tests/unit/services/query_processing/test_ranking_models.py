"""Model tests for the personalized ranking service.

Tests for interaction events, user profiles and ranked results.
"""

import asyncio
from datetime import UTC, datetime, timedelta
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
