"""Service calculation tests for the personalized ranking service.

Tests for boost calculations, scoring algorithms, and performance tracking.
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


class TestPersonalizedRankingServiceCalculations:
    """Test PersonalizedRankingService calculation algorithms."""

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
            _total_interactions=50,
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
        assert features["user__total_interactions"] == 50.0
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
        assert initial_stats["_total_rankings"] == 0

        # Update performance stats
        service._update_performance_stats(RankingStrategy.HYBRID, 150.0, True)
        service._update_performance_stats(RankingStrategy.DEFAULT, 50.0, False)
        service._update_performance_stats(RankingStrategy.HYBRID, 200.0, True)

        stats = service.get_performance_stats()
        assert stats["_total_rankings"] == 3
        assert stats["avg_processing_time"] == (150.0 + 50.0 + 200.0) / 3
        assert stats["personalization_rate"] == 2.0 / 3.0  # 2 personalized out of 3
        assert stats["strategy_usage"]["hybrid"] == 2
        assert stats["strategy_usage"]["default"] == 1

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
