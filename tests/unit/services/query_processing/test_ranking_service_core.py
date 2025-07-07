"""Core service tests for the personalized ranking service.

Tests for service initialization, profile management, and basic ranking functionality.
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


class TestPersonalizedRankingServiceCore:
    """Test PersonalizedRankingService core functionality."""

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
        assert stats["_total_rankings"] == 0
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
        assert profile._total_interactions == 0
        assert profile.confidence_score == 0.0
        assert "new_user" in service.user_profiles

    @pytest.mark.asyncio
    async def test_get_user_profile_existing_user(self, service):
        """Test getting profile for existing user."""
        # Create initial profile
        profile1 = await service._get_user_profile("existing_user")
        profile1.preferences.append(
            UserPreference(attribute="language", value="python", weight=0.8)
        )

        # Get same profile again
        profile2 = await service._get_user_profile("existing_user")

        assert profile1 is profile2
        assert len(profile2.preferences) == 1
        assert profile2.preferences[0].value == "python"

    @pytest.mark.asyncio
    async def test_basic_ranking_no_personalization(self, service, sample_request):
        """Test basic ranking without personalization."""
        # Disable personalization for this test
        sample_request.personalization_strength = 0.0

        result = await service.rank_results(sample_request)

        assert isinstance(result, PersonalizedRankingResult)
        assert len(result.ranked_results) == 3
        assert result.personalization_applied is False
        assert result.user_profile_confidence == 0.0

        # Should maintain original score order since no personalization
        assert (
            result.ranked_results[0].original_score
            >= result.ranked_results[1].original_score
        )
        assert (
            result.ranked_results[1].original_score
            >= result.ranked_results[2].original_score
        )

    @pytest.mark.asyncio
    async def test_ranking_strategy_selection(
        self, service, sample_request, sample_results
    ):
        """Test different ranking strategies."""
        # Test content-based strategy
        sample_request.strategy = RankingStrategy.CONTENT_BASED
        result_content = await service.rank_results(sample_request)
        assert result_content.strategy_used == RankingStrategy.CONTENT_BASED

        # Test collaborative filtering strategy
        sample_request.strategy = RankingStrategy.COLLABORATIVE_FILTERING
        result_collab = await service.rank_results(sample_request)
        assert result_collab.strategy_used == RankingStrategy.COLLABORATIVE_FILTERING

        # Test hybrid strategy
        sample_request.strategy = RankingStrategy.HYBRID
        result_hybrid = await service.rank_results(sample_request)
        assert result_hybrid.strategy_used == RankingStrategy.HYBRID

    @pytest.mark.asyncio
    async def test_ranking_with_user_preferences(self, service, sample_request):
        """Test ranking with user preferences."""
        # Set up user profile with preferences
        profile = await service._get_user_profile("user123")
        profile.preferences = [
            UserPreference(
                attribute="content_type", value="tutorial", weight=0.8, confidence=0.9
            ),
            UserPreference(
                attribute="language", value="python", weight=0.7, confidence=0.8
            ),
        ]
        profile.confidence_score = 0.85

        result = await service.rank_results(sample_request)

        assert result.personalization_applied is True
        assert result.user_profile_confidence == 0.85

        # Tutorial content should be boosted
        tutorial_results = [
            r
            for r in result.ranked_results
            if "Tutorial" in r.title or "Guide" in r.title
        ]
        assert len(tutorial_results) >= 2

    @pytest.mark.asyncio
    async def test_interaction_recording(self, service):
        """Test recording user interactions."""
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

        await service.record_interaction(interaction)

        assert "user123" in service.interaction_history
        assert len(service.interaction_history["user123"]) == 1
        assert service.interaction_history["user123"][0].result_id == "result789"

    @pytest.mark.asyncio
    async def test_interaction_without_query(self, service):
        """Test interaction recording without query."""
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
