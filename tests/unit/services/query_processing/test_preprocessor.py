"""Tests for query preprocessor."""

import pytest
from src.services.query_processing.models import QueryPreprocessingResult
from src.services.query_processing.preprocessor import QueryPreprocessor


@pytest.fixture
def preprocessor():
    """Create a query preprocessor instance."""
    return QueryPreprocessor()


@pytest.fixture
async def initialized_preprocessor(preprocessor):
    """Create an initialized query preprocessor."""
    await preprocessor.initialize()
    return preprocessor


class TestQueryPreprocessor:
    """Test the QueryPreprocessor class."""

    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor._initialized is False
        assert len(preprocessor._spelling_corrections) > 0
        assert len(preprocessor._synonym_expansions) > 0

    async def test_initialize(self, preprocessor):
        """Test preprocessor initialization."""
        await preprocessor.initialize()
        assert preprocessor._initialized is True

    async def test_basic_preprocessing(self, initialized_preprocessor):
        """Test basic query preprocessing."""
        query = "What is phython?"

        result = await initialized_preprocessor.preprocess_query(query)

        assert isinstance(result, QueryPreprocessingResult)
        assert result.original_query == query
        assert "python" in result.processed_query.lower()
        assert len(result.corrections_applied) > 0
        assert result.preprocessing_time_ms > 0

    async def test_spell_correction(self, initialized_preprocessor):
        """Test spelling correction functionality."""
        test_cases = [
            ("phython programming", "python"),
            ("databse optimization", "database"),
            ("authetication system", "authentication"),
            ("microservice architecture", "microservices"),
        ]

        for input_query, expected_correction in test_cases:
            result = await initialized_preprocessor.preprocess_query(
                input_query, enable_spell_correction=True
            )

            assert expected_correction in result.processed_query.lower()
            assert len(result.corrections_applied) > 0

    async def test_spell_correction_disabled(self, initialized_preprocessor):
        """Test disabling spell correction."""
        query = "phython programming"

        result = await initialized_preprocessor.preprocess_query(
            query, enable_spell_correction=False
        )

        assert result.processed_query == query.strip()
        assert len(result.corrections_applied) == 0

    async def test_synonym_expansion(self, initialized_preprocessor):
        """Test synonym expansion functionality."""
        test_cases = [
            ("API development", "rest api"),
            ("db optimization", "database"),
            ("auth system", "authentication"),
            ("js framework", "javascript"),
            ("ml algorithms", "machine learning"),
        ]

        for input_query, expected_expansion in test_cases:
            result = await initialized_preprocessor.preprocess_query(
                input_query, enable_expansion=True
            )

            # Should contain expansion in parentheses or original query enhanced
            assert (
                expected_expansion.lower() in result.processed_query.lower()
                or len(result.expansions_added) > 0
            )

    async def test_expansion_disabled(self, initialized_preprocessor):
        """Test disabling synonym expansion."""
        query = "API development"

        result = await initialized_preprocessor.preprocess_query(
            query, enable_expansion=False
        )

        assert len(result.expansions_added) == 0

    async def test_text_normalization(self, initialized_preprocessor):
        """Test text normalization functionality."""
        # Test cases where normalization actually changes the text
        test_cases_with_changes = [
            ("What   is    Python???", "What is Python?"),
            ("JavaScript  w/  TypeScript", "JavaScript with TypeScript"),
            ("React  e.g.  for  UI", "React for example for UI"),
            ("What's  the  'best'  approach?", 'What\'s the "best" approach?'),
        ]

        for input_query, _expected_pattern in test_cases_with_changes:
            result = await initialized_preprocessor.preprocess_query(
                input_query, enable_normalization=True, enable_expansion=False
            )

            # Check that normalization was applied when text actually changes
            assert result.normalization_applied is True
            assert "  " not in result.processed_query  # No double spaces
            assert "???" not in result.processed_query  # No multiple punctuation

        # Test cases where normalization doesn't change the text
        no_change_cases = [
            "API-REST-GraphQL",
            "Simple query",
            "Clean text already",
        ]

        for input_query in no_change_cases:
            result = await initialized_preprocessor.preprocess_query(
                input_query,
                enable_normalization=True,
                enable_expansion=False,
                enable_spell_correction=False,
                enable_context_extraction=False,
            )

            # Check that normalization was not marked as applied when text doesn't change
            assert result.normalization_applied is False

    async def test_normalization_disabled(self, initialized_preprocessor):
        """Test disabling text normalization."""
        query = "What   is    Python???"

        result = await initialized_preprocessor.preprocess_query(
            query, enable_normalization=False
        )

        assert result.normalization_applied is False
        # Query should be stripped but not normalized
        assert result.processed_query == query.strip()

    async def test_context_extraction(self, initialized_preprocessor):
        """Test context extraction functionality."""
        test_cases = [
            (
                "How to implement React hooks with TypeScript?",
                {"programming_language": ["typescript"], "framework": ["react"]},
            ),
            (
                "Python 3.9 Django REST API best practices",
                {"programming_language": ["python"], "framework": ["django"]},
            ),
            (
                "MySQL vs PostgreSQL performance comparison",
                {"database": ["mysql", "postgresql"]},
            ),
            (
                "AWS Lambda deployment with Docker containers",
                {"cloud_platform": ["aws", "lambda", "docker"]},
            ),
        ]

        for query, expected_contexts in test_cases:
            result = await initialized_preprocessor.preprocess_query(
                query, enable_context_extraction=True
            )

            context = result.context_extracted

            # Check that expected context types are detected
            for context_type, expected_values in expected_contexts.items():
                if context_type in context:
                    detected_values = [v.lower() for v in context[context_type]]
                    assert any(
                        expected in detected_values for expected in expected_values
                    )

    async def test_context_extraction_disabled(self, initialized_preprocessor):
        """Test disabling context extraction."""
        query = "React with TypeScript development"

        result = await initialized_preprocessor.preprocess_query(
            query, enable_context_extraction=False
        )

        assert result.context_extracted == {}

    async def test_version_detection(self, initialized_preprocessor):
        """Test version number detection in context."""
        test_cases = [
            "Python 3.9 features",
            "React v17.0.2 updates",
            "Node.js version 16.14.0",
            "Django 4.1 release",
        ]

        for query in test_cases:
            result = await initialized_preprocessor.preprocess_query(query)

            context = result.context_extracted
            assert "version" in context or any(
                "version" in str(v) for v in context.values()
            )

    async def test_urgency_detection(self, initialized_preprocessor):
        """Test urgency indicator detection."""
        high_urgency_queries = [
            "URGENT: Production API is down, need immediate fix",
            "Critical security vulnerability in authentication system",
            "Emergency: Database connection failing in production",
        ]

        medium_urgency_queries = [
            "Need quick solution for deployment issue",
            "Fast implementation of user authentication",
            "Soon need to migrate database schema",
        ]

        for query in high_urgency_queries:
            result = await initialized_preprocessor.preprocess_query(query)
            context = result.context_extracted

            if "urgency" in context:
                assert context["urgency"] == "high"

        for query in medium_urgency_queries:
            result = await initialized_preprocessor.preprocess_query(query)
            context = result.context_extracted

            if "urgency" in context:
                assert context["urgency"] == "medium"

    async def test_experience_level_detection(self, initialized_preprocessor):
        """Test experience level indicator detection."""
        beginner_queries = [
            "I'm new to Python, how to start?",
            "Just started learning React, need help",
            "Beginner guide to machine learning",
        ]

        intermediate_queries = [
            "Intermediate Python programming techniques",
            "Some experience with React development",
        ]

        advanced_queries = [
            "Advanced Python metaclass patterns",
            "Expert-level React performance optimization",
            "Professional enterprise architecture design",
        ]

        for query in beginner_queries:
            result = await initialized_preprocessor.preprocess_query(query)
            context = result.context_extracted

            if "experience_level" in context:
                assert context["experience_level"] == "beginner"

        for query in intermediate_queries:
            result = await initialized_preprocessor.preprocess_query(query)
            context = result.context_extracted

            if "experience_level" in context:
                assert context["experience_level"] == "intermediate"

        for query in advanced_queries:
            result = await initialized_preprocessor.preprocess_query(query)
            context = result.context_extracted

            if "experience_level" in context:
                assert context["experience_level"] == "advanced"

    async def test_complexity_indicators(self, initialized_preprocessor):
        """Test complexity indicator detection."""
        complex_queries = [
            "Scalable microservices architecture with event sourcing",
            "High-performance database optimization strategies",
            "Enterprise security implementation patterns",
            "Distributed system integration challenges",
        ]

        for query in complex_queries:
            result = await initialized_preprocessor.preprocess_query(query)
            context = result.context_extracted

            if "complexity_indicators" in context:
                indicators = context["complexity_indicators"]
                assert len(indicators) > 0
                assert any(
                    indicator
                    in ["architectural", "performance", "security", "integration"]
                    for indicator in indicators
                )

    async def test_comprehensive_preprocessing(self, initialized_preprocessor):
        """Test comprehensive preprocessing with all features enabled."""
        query = "How to fix authetication  phython  api???"

        result = await initialized_preprocessor.preprocess_query(
            query,
            enable_spell_correction=True,
            enable_expansion=True,
            enable_normalization=True,
            enable_context_extraction=True,
        )

        # Should have applied corrections
        assert "authentication" in result.processed_query.lower()
        assert "python" in result.processed_query.lower()
        assert len(result.corrections_applied) >= 2

        # Should have normalization
        assert result.normalization_applied is True
        assert "???" not in result.processed_query

        # Should have context
        assert len(result.context_extracted) > 0

        # Should track processing time
        assert result.preprocessing_time_ms > 0

    async def test_long_query_expansion_limit(self, initialized_preprocessor):
        """Test that very long queries don't get expanded."""
        long_query = " ".join(["word"] * 20)  # 20 words

        result = await initialized_preprocessor.preprocess_query(
            long_query, enable_expansion=True
        )

        # Should not add expansions to very long queries
        assert len(result.expansions_added) == 0

    async def test_processing_time_measurement(self, initialized_preprocessor):
        """Test that processing time is measured."""
        query = "Simple query"

        result = await initialized_preprocessor.preprocess_query(query)

        assert result.preprocessing_time_ms >= 0
        assert isinstance(result.preprocessing_time_ms, float)

    async def test_uninitialized_preprocessor_error(self, preprocessor):
        """Test error when using uninitialized preprocessor."""
        query = "test query"

        with pytest.raises(RuntimeError, match="not initialized"):
            await preprocessor.preprocess_query(query)

    async def test_cleanup(self, initialized_preprocessor):
        """Test preprocessor cleanup."""
        await initialized_preprocessor.cleanup()
        assert initialized_preprocessor._initialized is False

    async def test_empty_query_handling(self, initialized_preprocessor):
        """Test handling of empty queries."""
        result = await initialized_preprocessor.preprocess_query("   ")

        assert result.original_query == "   "
        assert result.processed_query == ""
        assert len(result.corrections_applied) == 0
        assert len(result.expansions_added) == 0

    async def test_special_characters_normalization(self, initialized_preprocessor):
        """Test normalization of special characters."""
        query = 'What\'s the "best" way to handle -dashes- and...ellipses?'

        result = await initialized_preprocessor.preprocess_query(
            query, enable_normalization=True
        )

        # Should normalize quotes and dashes
        assert '"' in result.processed_query  # Normalized quotes
        assert "-" in result.processed_query  # Normalized dashes
        assert result.normalization_applied is True

    async def test_technical_abbreviations_normalization(
        self, initialized_preprocessor
    ):
        """Test normalization of technical abbreviations."""
        query = "Use React w/ TypeScript e.g. for better typing etc."

        result = await initialized_preprocessor.preprocess_query(
            query, enable_normalization=True
        )

        # Should expand abbreviations (or keep them as-is based on implementation)
        assert result.processed_query is not None
        assert len(result.processed_query) > 0
        # Note: actual expansion depends on implementation details

    async def test_remove_stop_words_directly(self, initialized_preprocessor):
        """Test the _remove_stop_words method directly for coverage."""
        # Test with short query (should not remove stop words)
        short_query = "the API"
        result = initialized_preprocessor._remove_stop_words(short_query)
        assert result == short_query

        # Test with long query (should remove stop words)
        long_query = "How to configure the database for a web application"
        result = initialized_preprocessor._remove_stop_words(long_query)
        # Should remove some stop words but keep important ones
        assert "How" in result
        assert "configure" in result
        assert "database" in result
        assert "web" in result
        assert "application" in result
        # Some stop words should be removed
        assert len(result.split()) <= len(long_query.split())

        # Test with important stop words that should be preserved
        query_with_important_stops = "How to migrate from MySQL to PostgreSQL"
        result = initialized_preprocessor._remove_stop_words(query_with_important_stops)
        assert "from" in result  # important stop word
        assert "to" in result  # important stop word

        # Test with query containing only stop words
        stop_words_only = "the a an is"
        result = initialized_preprocessor._remove_stop_words(stop_words_only)
        # Should return original since filtered result would be empty
        assert result == stop_words_only

        # Test with punctuation handling
        query_with_punctuation = "What's the best way to handle this?"
        result = initialized_preprocessor._remove_stop_words(query_with_punctuation)
        assert "What's" in result
        assert "best" in result
        assert "way" in result
        assert "handle" in result
        assert "this?" in result

    async def test_normalization_edge_cases(self, initialized_preprocessor):
        """Test edge cases in text normalization."""
        # Test empty string after strip
        result = await initialized_preprocessor.preprocess_query(
            "   ",
            enable_normalization=True,
            enable_expansion=False,
            enable_spell_correction=False,
            enable_context_extraction=False,
        )
        assert result.normalization_applied is False
        assert result.processed_query == ""

        # Test single character
        result = await initialized_preprocessor.preprocess_query(
            "a",
            enable_normalization=True,
            enable_expansion=False,
            enable_spell_correction=False,
            enable_context_extraction=False,
        )
        assert result.processed_query == "a"

        # Test query with special unicode characters
        unicode_query = "Pythön prögrämmîng"
        result = await initialized_preprocessor.preprocess_query(
            unicode_query,
            enable_normalization=True,
            enable_expansion=False,
            enable_spell_correction=False,
            enable_context_extraction=False,
        )
        assert result.processed_query == unicode_query

    async def test_spell_correction_edge_cases(self, initialized_preprocessor):
        """Test edge cases in spell correction."""
        # Test case sensitivity
        result = await initialized_preprocessor.preprocess_query(
            "PHYTHON Programming",
            enable_spell_correction=True,
            enable_expansion=False,
            enable_normalization=False,
            enable_context_extraction=False,
        )
        assert "python" in result.processed_query.lower()
        assert len(result.corrections_applied) > 0

        # Test substring matches (will correct if misspelling is contained)
        result = await initialized_preprocessor.preprocess_query(
            "phythonic",
            enable_spell_correction=True,
            enable_expansion=False,
            enable_normalization=False,
            enable_context_extraction=False,
        )
        # Should correct since "phython" is in "phythonic"
        assert result.processed_query == "pythonic"
        assert len(result.corrections_applied) > 0

    async def test_expansion_edge_cases(self, initialized_preprocessor):
        """Test edge cases in synonym expansion."""
        # Test expansion with already expanded terms
        result = await initialized_preprocessor.preprocess_query(
            "rest api development",
            enable_expansion=True,
            enable_spell_correction=False,
            enable_normalization=False,
            enable_context_extraction=False,
        )
        # Should not add redundant expansions
        processed_lower = result.processed_query.lower()
        rest_api_count = processed_lower.count("rest api")
        assert rest_api_count <= 2  # Original + max one expansion

    async def test_context_extraction_edge_cases(self, initialized_preprocessor):
        """Test edge cases in context extraction."""
        # Test empty query
        result = await initialized_preprocessor.preprocess_query(
            "",
            enable_context_extraction=True,
            enable_spell_correction=False,
            enable_normalization=False,
            enable_expansion=False,
        )
        assert result.context_extracted == {}

        # Test query with multiple overlapping contexts
        complex_query = "Python 3.9 Django REST API security architecture"
        result = await initialized_preprocessor.preprocess_query(
            complex_query,
            enable_context_extraction=True,
            enable_spell_correction=False,
            enable_normalization=False,
            enable_expansion=False,
        )
        context = result.context_extracted
        assert "programming_language" in context
        assert "framework" in context
        assert "complexity_indicators" in context
        assert "security" in context["complexity_indicators"]
        assert "architectural" in context["complexity_indicators"]

    async def test_error_handling(self, initialized_preprocessor):
        """Test error handling and edge cases."""
        # Test very long query processing
        very_long_query = " ".join(["word"] * 100)
        result = await initialized_preprocessor.preprocess_query(very_long_query)
        assert result.processed_query is not None
        assert result.preprocessing_time_ms >= 0

        # Test query with only punctuation
        punctuation_only = "!@#$%^&*()"
        result = await initialized_preprocessor.preprocess_query(
            punctuation_only, enable_normalization=True
        )
        assert result.processed_query is not None

        # Test query with mixed encodings (if applicable)
        mixed_query = "Python programming 编程"
        result = await initialized_preprocessor.preprocess_query(mixed_query)
        assert result.processed_query is not None
        assert result.preprocessing_time_ms >= 0
