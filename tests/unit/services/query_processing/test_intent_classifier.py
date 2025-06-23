"""Tests for query intent classifier."""

from unittest.mock import AsyncMock

import pytest

from src.services.query_processing.intent_classifier import QueryIntentClassifier
from src.services.query_processing.models import QueryComplexity
from src.services.query_processing.models import QueryIntent
from src.services.query_processing.models import QueryIntentClassification


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager."""
    manager = AsyncMock()
    # Return embeddings for query + 14 reference embeddings (one per intent)
    embeddings = [[0.1] * 768]  # Query embedding
    embeddings.extend(
        [[0.2 + i * 0.01] * 768 for i in range(14)]
    )  # 14 reference embeddings

    manager.generate_embeddings = AsyncMock(
        return_value={"success": True, "embeddings": embeddings}
    )
    return manager


@pytest.fixture
def intent_classifier(mock_embedding_manager):
    """Create an intent classifier instance."""
    return QueryIntentClassifier(mock_embedding_manager)


@pytest.fixture
async def initialized_classifier(intent_classifier):
    """Create an initialized intent classifier."""
    await intent_classifier.initialize()
    return intent_classifier


class TestQueryIntentClassifier:
    """Test the QueryIntentClassifier class."""

    def test_initialization(self, intent_classifier):
        """Test classifier initialization."""
        assert intent_classifier.embedding_manager is not None
        assert intent_classifier._initialized is False

    async def test_initialize(self, intent_classifier):
        """Test classifier initialization."""
        await intent_classifier.initialize()
        assert intent_classifier._initialized is True

    async def test_classify_conceptual_query(self, initialized_classifier):
        """Test classifying a conceptual query."""
        query = "What is machine learning and how does it work?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert isinstance(result, QueryIntentClassification)
        assert result.primary_intent == QueryIntent.CONCEPTUAL
        assert result.confidence_scores[QueryIntent.CONCEPTUAL] > 0.0
        assert result.complexity_level in [
            QueryComplexity.SIMPLE,
            QueryComplexity.MODERATE,
            QueryComplexity.COMPLEX,
            QueryComplexity.EXPERT,
        ]

    async def test_classify_procedural_query(self, initialized_classifier):
        """Test classifying a procedural query."""
        query = "How to implement authentication in Python step by step?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.primary_intent == QueryIntent.PROCEDURAL
        assert result.confidence_scores[QueryIntent.PROCEDURAL] > 0.0

    async def test_classify_factual_query(self, initialized_classifier):
        """Test classifying a factual query."""
        query = "What version of Python supports async/await?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.primary_intent == QueryIntent.FACTUAL
        assert result.confidence_scores[QueryIntent.FACTUAL] > 0.0

    async def test_classify_troubleshooting_query(self, initialized_classifier):
        """Test classifying a troubleshooting query."""
        query = "Getting ImportError when trying to import pandas, how to fix?"

        result = await initialized_classifier.classify_query_advanced(query)

        # Should detect troubleshooting intent (may be primary or secondary)
        all_intents = [result.primary_intent, *result.secondary_intents]
        assert QueryIntent.TROUBLESHOOTING in all_intents
        assert result.confidence_scores[QueryIntent.TROUBLESHOOTING] > 0.0

    async def test_classify_comparative_query(self, initialized_classifier):
        """Test classifying a comparative query."""
        query = "React vs Vue vs Angular - which is better for large projects?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.primary_intent == QueryIntent.COMPARATIVE
        assert result.confidence_scores[QueryIntent.COMPARATIVE] > 0.0

    async def test_classify_architectural_query(self, initialized_classifier):
        """Test classifying an architectural query."""
        query = "How to design a scalable microservices architecture?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.primary_intent == QueryIntent.ARCHITECTURAL
        assert result.confidence_scores[QueryIntent.ARCHITECTURAL] > 0.0

    async def test_classify_performance_query(self, initialized_classifier):
        """Test classifying a performance query."""
        query = "How to optimize database query performance and reduce latency?"

        result = await initialized_classifier.classify_query_advanced(query)

        # Should detect performance intent (may be primary or secondary)
        all_intents = [result.primary_intent, *result.secondary_intents]
        assert QueryIntent.PERFORMANCE in all_intents
        assert result.confidence_scores[QueryIntent.PERFORMANCE] > 0.0

    async def test_classify_security_query(self, initialized_classifier):
        """Test classifying a security query."""
        query = "How to implement OAuth 2.0 authentication securely?"

        result = await initialized_classifier.classify_query_advanced(query)

        # Should detect security intent (may be primary or secondary)
        all_intents = [result.primary_intent, *result.secondary_intents]
        assert QueryIntent.SECURITY in all_intents
        assert result.confidence_scores[QueryIntent.SECURITY] > 0.0

    async def test_classify_integration_query(self, initialized_classifier):
        """Test classifying an integration query."""
        query = "How to integrate with third-party REST API using webhooks?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.primary_intent == QueryIntent.INTEGRATION
        assert result.confidence_scores[QueryIntent.INTEGRATION] > 0.0

    async def test_classify_best_practices_query(self, initialized_classifier):
        """Test classifying a best practices query."""
        query = "What are the best practices for Python code organization?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.primary_intent == QueryIntent.BEST_PRACTICES
        assert result.confidence_scores[QueryIntent.BEST_PRACTICES] > 0.0

    async def test_classify_code_review_query(self, initialized_classifier):
        """Test classifying a code review query."""
        query = "Please review my Python code and suggest improvements"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.primary_intent == QueryIntent.CODE_REVIEW
        assert result.confidence_scores[QueryIntent.CODE_REVIEW] > 0.0

    async def test_classify_migration_query(self, initialized_classifier):
        """Test classifying a migration query."""
        query = "How to migrate from Python 2.7 to Python 3.9?"

        result = await initialized_classifier.classify_query_advanced(query)

        # Should detect migration intent (may be primary or secondary)
        all_intents = [result.primary_intent, *result.secondary_intents]
        assert QueryIntent.MIGRATION in all_intents
        assert result.confidence_scores[QueryIntent.MIGRATION] > 0.0

    async def test_classify_debugging_query(self, initialized_classifier):
        """Test classifying a debugging query."""
        query = "How to debug memory leaks using profiling tools?"

        result = await initialized_classifier.classify_query_advanced(query)

        # Should detect debugging intent (may be primary or secondary)
        all_intents = [result.primary_intent, *result.secondary_intents]
        assert QueryIntent.DEBUGGING in all_intents
        assert result.confidence_scores[QueryIntent.DEBUGGING] > 0.0

    async def test_classify_configuration_query(self, initialized_classifier):
        """Test classifying a configuration query."""
        query = "How to configure Django settings for production environment?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.primary_intent == QueryIntent.CONFIGURATION
        assert result.confidence_scores[QueryIntent.CONFIGURATION] > 0.0

    async def test_secondary_intents(self, initialized_classifier):
        """Test detection of secondary intents."""
        query = "How to optimize React performance and debug rendering issues?"

        result = await initialized_classifier.classify_query_advanced(query)

        # Should detect both performance and debugging intents
        assert len(result.secondary_intents) > 0
        assert any(
            intent in [QueryIntent.PERFORMANCE, QueryIntent.DEBUGGING]
            for intent in result.secondary_intents
        )

    async def test_complexity_assessment(self, initialized_classifier):
        """Test query complexity assessment."""
        # Simple query
        simple_result = await initialized_classifier.classify_query_advanced(
            "What is Python?"
        )
        assert simple_result.complexity_level in [
            QueryComplexity.SIMPLE,
            QueryComplexity.MODERATE,
            QueryComplexity.COMPLEX,
            QueryComplexity.EXPERT,
        ]

        # Complex query
        complex_query = (
            "How to design a distributed microservices architecture "
            "with event sourcing, CQRS, and saga patterns for handling "
            "cross-service transactions in a high-throughput system?"
        )
        complex_result = await initialized_classifier.classify_query_advanced(
            complex_query
        )
        assert complex_result.complexity_level in [
            QueryComplexity.COMPLEX,
            QueryComplexity.EXPERT,
        ]

    async def test_domain_detection(self, initialized_classifier):
        """Test technical domain detection."""
        query = "How to implement React hooks with TypeScript?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.domain_category is not None
        assert result.domain_category in ["web_development", "backend", "mobile"]

    async def test_context_extraction(self, initialized_classifier):
        """Test context information extraction."""
        query = "How to fix Python 3.9 ImportError in Django production?"
        context = {"framework": ["django"], "urgency": "high"}

        result = await initialized_classifier.classify_query_advanced(query, context)

        assert result.classification_reasoning is not None
        assert len(result.classification_reasoning) > 0

    async def test_suggested_followups(self, initialized_classifier):
        """Test generation of suggested follow-up questions."""
        query = "What is Docker?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert len(result.suggested_followups) > 0
        assert all(isinstance(followup, str) for followup in result.suggested_followups)

    async def test_requires_context_detection(self, initialized_classifier):
        """Test detection of queries that require additional context."""
        # Vague query should require context
        vague_result = await initialized_classifier.classify_query_advanced("Fix it")
        assert vague_result.requires_context is True

        # Complex architectural query should require context
        arch_query = "Design microservices architecture"
        arch_result = await initialized_classifier.classify_query_advanced(arch_query)
        assert arch_result.requires_context is True

    async def test_empty_query_handling(self, initialized_classifier):
        """Test handling of empty or whitespace-only queries."""
        result = await initialized_classifier.classify_query_advanced("   ")

        assert result.primary_intent == QueryIntent.FACTUAL  # Default fallback
        assert result.confidence_scores[QueryIntent.FACTUAL] < 0.5

    async def test_semantic_classification_fallback(self, intent_classifier):
        """Test fallback when semantic classification fails."""
        # Mock embedding manager to fail
        intent_classifier.embedding_manager.generate_embeddings = AsyncMock(
            side_effect=Exception("Embedding failed")
        )
        await intent_classifier.initialize()

        query = "What is machine learning?"
        result = await intent_classifier.classify_query_advanced(query)

        # Should still work with rule-based classification
        assert result.primary_intent == QueryIntent.CONCEPTUAL
        assert result.confidence_scores[QueryIntent.CONCEPTUAL] > 0.0

    async def test_uninitialized_classifier_error(self, intent_classifier):
        """Test error when using uninitialized classifier."""
        query = "test query"

        with pytest.raises(RuntimeError, match="not initialized"):
            await intent_classifier.classify_query_advanced(query)

    async def test_confidence_score_validation(self, initialized_classifier):
        """Test that confidence scores are within valid range."""
        query = "How to implement machine learning algorithms in Python?"

        result = await initialized_classifier.classify_query_advanced(query)

        # All confidence scores should be between 0 and 1
        for score in result.confidence_scores.values():
            assert 0.0 <= score <= 1.0

    async def test_classification_reasoning_content(self, initialized_classifier):
        """Test that classification reasoning contains useful information."""
        query = "What are the differences between SQL and NoSQL databases?"

        result = await initialized_classifier.classify_query_advanced(query)

        assert result.classification_reasoning is not None
        assert len(result.classification_reasoning) > 10  # Should be descriptive
        assert ":" in result.classification_reasoning  # Should contain scores

    async def test_cleanup(self, initialized_classifier):
        """Test classifier cleanup."""
        await initialized_classifier.cleanup()
        assert initialized_classifier._initialized is False
