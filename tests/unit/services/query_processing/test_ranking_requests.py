"""Request and result tests for the personalized ranking service.

Tests for PersonalizedRankingRequest and PersonalizedRankingResult classes.
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
