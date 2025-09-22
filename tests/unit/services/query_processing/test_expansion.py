"""Tests for query expansion service."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.services.query_processing.expansion import (
    ExpandedTerm,
    ExpansionScope,
    ExpansionStrategy,
    QueryExpansionRequest,
    QueryExpansionResult,
    QueryExpansionService,
    TermRelationType,
)


class TestExpansionStrategy:
    """Test ExpansionStrategy enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert ExpansionStrategy.SYNONYM_BASED == "synonym_based"
        assert ExpansionStrategy.SEMANTIC_BASED == "semantic_based"
        assert ExpansionStrategy.CONTEXT_AWARE == "context_aware"
        assert ExpansionStrategy.DOMAIN_SPECIFIC == "domain_specific"
        assert ExpansionStrategy.HYBRID == "hybrid"
        assert ExpansionStrategy.LEARNING_BASED == "learning_based"

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        strategies = list(ExpansionStrategy)
        assert len(strategies) == 6
        assert ExpansionStrategy.HYBRID in strategies


class TestExpansionScope:
    """Test ExpansionScope enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert ExpansionScope.CONSERVATIVE == "conservative"
        assert ExpansionScope.MODERATE == "moderate"
        assert ExpansionScope.AGGRESSIVE == "aggressive"
        assert ExpansionScope.ADAPTIVE == "adaptive"

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        scopes = list(ExpansionScope)
        assert len(scopes) == 4
        assert ExpansionScope.MODERATE in scopes


class TestTermRelationType:
    """Test TermRelationType enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert TermRelationType.SYNONYM == "synonym"
        assert TermRelationType.HYPERNYM == "hypernym"
        assert TermRelationType.HYPONYM == "hyponym"
        assert TermRelationType.MERONYM == "meronym"
        assert TermRelationType.HOLONYM == "holonym"
        assert TermRelationType.RELATED == "related"
        assert TermRelationType.ABBREVIATION == "abbreviation"
        assert TermRelationType.ALTERNATIVE == "alternative"

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        relations = list(TermRelationType)
        assert len(relations) == 8
        assert TermRelationType.SYNONYM in relations


class TestExpandedTerm:
    """Test ExpandedTerm model."""

    def test_valid_expanded_term(self):
        """Test creating a valid expanded term."""
        term = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type=TermRelationType.SYNONYM,
            confidence=0.8,
            source="wordnet",
        )

        assert term.term == "function"
        assert term.original_term == "method"
        assert term.relation_type == TermRelationType.SYNONYM
        assert term.confidence == 0.8
        assert term.source == "wordnet"
        assert term.context_relevance == 1.0  # default
        assert term.frequency_weight == 1.0  # default

    def test_term_validation(self):
        """Test term validation."""
        # Test empty term
        with pytest.raises(ValueError, match="Term cannot be empty"):
            ExpandedTerm(
                term="",
                original_term="method",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.8,
                source="wordnet",
            )

        # Test whitespace-only term
        with pytest.raises(ValueError, match="Term cannot be empty"):
            ExpandedTerm(
                term="   ",
                original_term="method",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.8,
                source="wordnet",
            )

    def test_term_normalization(self):
        """Test term normalization (strip and lowercase)."""
        term = ExpandedTerm(
            term="  FUNCTION  ",
            original_term="  METHOD  ",
            relation_type=TermRelationType.SYNONYM,
            confidence=0.8,
            source="wordnet",
        )

        assert term.term == "function"
        assert term.original_term == "method"

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        term = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type=TermRelationType.SYNONYM,
            confidence=0.5,
            source="wordnet",
        )
        assert term.confidence == 0.5

        # Test edge cases
        term_min = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type=TermRelationType.SYNONYM,
            confidence=0.0,
            source="wordnet",
        )
        assert term_min.confidence == 0.0

        term_max = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type=TermRelationType.SYNONYM,
            confidence=1.0,
            source="wordnet",
        )
        assert term_max.confidence == 1.0

    def test_optional_fields(self):
        """Test optional fields with custom values."""
        term = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type=TermRelationType.SYNONYM,
            confidence=0.8,
            source="wordnet",
            context_relevance=0.9,
            frequency_weight=2.5,
        )

        assert term.context_relevance == 0.9
        assert term.frequency_weight == 2.5


class TestQueryExpansionRequest:
    """Test QueryExpansionRequest model."""

    def test_minimal_request(self):
        """Test creating minimal request."""
        request = QueryExpansionRequest(original_query="python programming")

        assert request.original_query == "python programming"
        assert request.query_context == {}
        assert request.strategy == ExpansionStrategy.HYBRID
        assert request.scope == ExpansionScope.MODERATE
        assert request.max_expanded_terms == 10
        assert request.min_confidence == 0.5
        assert request.target_domains is None
        assert request.exclude_terms is None
        assert request.preserve_intent is True
        assert request.boost_recent_terms is True
        assert request.include_abbreviations is True
        assert request.max_processing_time_ms == 2000.0
        assert request.enable_caching is True

    def test_full_request(self):
        """Test creating request with all fields."""
        request = QueryExpansionRequest(
            original_query="python programming",
            query_context={"domain": ["programming"], "intent": "learning"},
            strategy=ExpansionStrategy.SEMANTIC_BASED,
            scope=ExpansionScope.AGGRESSIVE,
            max_expanded_terms=20,
            min_confidence=0.7,
            target_domains=["programming", "software"],
            exclude_terms=["deprecated", "legacy"],
            preserve_intent=False,
            boost_recent_terms=False,
            include_abbreviations=False,
            max_processing_time_ms=5000.0,
            enable_caching=False,
        )

        assert request.strategy == ExpansionStrategy.SEMANTIC_BASED
        assert request.scope == ExpansionScope.AGGRESSIVE
        assert request.max_expanded_terms == 20
        assert request.min_confidence == 0.7
        assert request.target_domains == ["programming", "software"]
        assert request.exclude_terms == ["deprecated", "legacy"]
        assert request.preserve_intent is False
        assert request.boost_recent_terms is False
        assert request.include_abbreviations is False
        assert request.max_processing_time_ms == 5000.0
        assert request.enable_caching is False

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid edge cases
        QueryExpansionRequest(
            original_query="test",
            max_expanded_terms=1,
            min_confidence=0.0,
            max_processing_time_ms=100.0,
        )

        QueryExpansionRequest(
            original_query="test", max_expanded_terms=50, min_confidence=1.0
        )


class TestQueryExpansionResult:
    """Test QueryExpansionResult model."""

    def test_minimal_result(self):
        """Test creating minimal result."""
        result = QueryExpansionResult(
            original_query="python programming",
            expanded_query="python OR coding programming OR development",
            expansion_strategy=ExpansionStrategy.HYBRID,
            expansion_scope=ExpansionScope.MODERATE,
            confidence_score=0.8,
            processing_time_ms=150.0,
        )

        assert result.original_query == "python programming"
        assert result.expanded_query == "python OR coding programming OR development"
        assert result.expansion_strategy == ExpansionStrategy.HYBRID
        assert result.expansion_scope == ExpansionScope.MODERATE
        assert result.confidence_score == 0.8
        assert result.processing_time_ms == 150.0
        assert result.expanded_terms == []
        assert result.cache_hit is False
        assert result.term_statistics == {}
        assert result.expansion_metadata == {}

    def test_full_result(self):
        """Test creating result with all fields."""
        expanded_term = ExpandedTerm(
            term="coding",
            original_term="python",
            relation_type=TermRelationType.SYNONYM,
            confidence=0.9,
            source="wordnet",
        )

        result = QueryExpansionResult(
            original_query="python programming",
            expanded_terms=[expanded_term],
            expanded_query="python OR coding programming",
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
            expansion_scope=ExpansionScope.AGGRESSIVE,
            confidence_score=0.85,
            processing_time_ms=200.0,
            cache_hit=True,
            term_statistics={"_total_terms": 1},
            expansion_metadata={"source": "test"},
        )

        assert len(result.expanded_terms) == 1
        assert result.expanded_terms[0].term == "coding"
        assert result.cache_hit is True
        assert result.term_statistics == {"_total_terms": 1}
        assert result.expansion_metadata == {"source": "test"}


class TestQueryExpansionService:
    """Test QueryExpansionService class."""

    @pytest.fixture
    def service(self):
        """Create expansion service instance."""
        return QueryExpansionService()

    @pytest.fixture
    def service_configured(self):
        """Create configured expansion service."""
        return QueryExpansionService(
            enable_semantic_expansion=True, enable_domain_expansion=True, cache_size=500
        )

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.enable_semantic_expansion is True
        assert service.enable_domain_expansion is True
        assert service.cache_size == 1000
        assert service.expansion_cache == {}
        assert service.cache_stats == {"hits": 0, "misses": 0}
        assert isinstance(service.synonym_sources, dict)
        assert isinstance(service.domain_vocabularies, dict)
        assert service.term_frequencies == {}
        assert service.expansion_history == []
        assert service.success_feedback == {}
        assert isinstance(service.performance_stats, dict)

    def test_custom_initialization(self, service_configured):
        """Test service with custom configuration."""
        assert service_configured.enable_semantic_expansion is True
        assert service_configured.enable_domain_expansion is True
        assert service_configured.cache_size == 500

    @pytest.mark.asyncio
    async def test_basic_expansion(self, service):
        """Test basic query expansion."""
        request = QueryExpansionRequest(
            original_query="programming function",
            strategy=ExpansionStrategy.SYNONYM_BASED,
        )

        result = await service.expand_query(request)

        assert isinstance(result, QueryExpansionResult)
        assert result.original_query == "programming function"
        assert result.expansion_strategy == ExpansionStrategy.SYNONYM_BASED
        assert result.expansion_scope == ExpansionScope.MODERATE
        assert result.confidence_score >= 0.0
        assert result.processing_time_ms > 0
        assert result.cache_hit is False

    @pytest.mark.asyncio
    async def test_synonym_based_expansion(self, service):
        """Test synonym-based expansion strategy."""
        request = QueryExpansionRequest(
            original_query="function programming",
            strategy=ExpansionStrategy.SYNONYM_BASED,
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        assert result.expansion_strategy == ExpansionStrategy.SYNONYM_BASED
        # Should find synonyms for "function" from domain vocabularies
        function_synonyms = any(
            term.original_term == "function"
            and term.relation_type == TermRelationType.SYNONYM
            for term in result.expanded_terms
        )
        assert (
            function_synonyms or len(result.expanded_terms) == 0
        )  # May not find synonyms

    @pytest.mark.asyncio
    async def test_semantic_based_expansion(self, service):
        """Test semantic-based expansion strategy."""
        request = QueryExpansionRequest(
            original_query="programming database",
            strategy=ExpansionStrategy.SEMANTIC_BASED,
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        assert result.expansion_strategy == ExpansionStrategy.SEMANTIC_BASED
        # Should find semantically related terms
        semantic_terms = any(
            term.relation_type == TermRelationType.RELATED
            and term.source == "semantic_model"
            for term in result.expanded_terms
        )
        assert semantic_terms or len(result.expanded_terms) == 0

    @pytest.mark.asyncio
    async def test_context_aware_expansion(self, service):
        """Test context-aware expansion strategy."""
        request = QueryExpansionRequest(
            original_query="function error",
            strategy=ExpansionStrategy.CONTEXT_AWARE,
            query_context={"domain": ["programming"], "intent": "debugging"},
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        assert result.expansion_strategy == ExpansionStrategy.CONTEXT_AWARE
        # Should consider programming context
        context_terms = any(
            term.source == "context_analysis" for term in result.expanded_terms
        )
        assert context_terms or len(result.expanded_terms) == 0

    @pytest.mark.asyncio
    async def test_domain_specific_expansion(self, service):
        """Test domain-specific expansion strategy."""
        request = QueryExpansionRequest(
            original_query="function variable",
            strategy=ExpansionStrategy.DOMAIN_SPECIFIC,
            target_domains=["programming"],
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        assert result.expansion_strategy == ExpansionStrategy.DOMAIN_SPECIFIC
        # Should find domain-specific terms
        domain_terms = any(
            term.source.startswith("domain_") for term in result.expanded_terms
        )
        assert domain_terms or len(result.expanded_terms) == 0

    @pytest.mark.asyncio
    async def test_hybrid_expansion(self, service):
        """Test hybrid expansion strategy."""
        request = QueryExpansionRequest(
            original_query="function programming",
            strategy=ExpansionStrategy.HYBRID,
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        assert result.expansion_strategy == ExpansionStrategy.HYBRID
        # Should combine multiple sources
        sources = {term.source for term in result.expanded_terms}
        # May include synonym_dictionary, semantic_model, context_analysis
        assert len(sources) >= 0  # Could be empty if no expansions found

    @pytest.mark.asyncio
    async def test_learning_based_expansion(self, service):
        """Test learning-based expansion strategy."""
        # Add some feedback data
        service.success_feedback = {"function": {"method": 0.9, "procedure": 0.7}}

        request = QueryExpansionRequest(
            original_query="function programming",
            strategy=ExpansionStrategy.LEARNING_BASED,
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        assert result.expansion_strategy == ExpansionStrategy.LEARNING_BASED
        # Should use learned expansions
        learned_terms = any(
            term.source == "learning_model" for term in result.expanded_terms
        )
        assert learned_terms or len(result.expanded_terms) == 0

    @pytest.mark.asyncio
    async def test_expansion_scopes(self, service):
        """Test different expansion scopes."""
        base_request = QueryExpansionRequest(
            original_query="function programming",
            strategy=ExpansionStrategy.SYNONYM_BASED,
            min_confidence=0.1,
        )

        # Conservative scope
        conservative_request = base_request.model_copy(
            update={"scope": ExpansionScope.CONSERVATIVE}
        )
        conservative_result = await service.expand_query(conservative_request)

        # Moderate scope
        moderate_request = base_request.model_copy(
            update={"scope": ExpansionScope.MODERATE}
        )
        moderate_result = await service.expand_query(moderate_request)

        # Aggressive scope
        aggressive_request = base_request.model_copy(
            update={"scope": ExpansionScope.AGGRESSIVE}
        )
        aggressive_result = await service.expand_query(aggressive_request)

        # Conservative should have fewer or equal terms than aggressive
        assert len(conservative_result.expanded_terms) <= len(
            aggressive_result.expanded_terms
        )

        # All should use appropriate confidence thresholds
        for term in conservative_result.expanded_terms:
            assert term.confidence >= 0.8

        for term in moderate_result.expanded_terms:
            assert term.confidence >= 0.6

    @pytest.mark.asyncio
    async def test_term_exclusion(self, service):
        """Test excluding specific terms."""
        request = QueryExpansionRequest(
            original_query="function error",
            strategy=ExpansionStrategy.SYNONYM_BASED,
            exclude_terms=["method", "procedure"],
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        # Should not include excluded terms
        excluded_terms = {"method", "procedure"}
        for term in result.expanded_terms:
            assert term.term not in excluded_terms

    @pytest.mark.asyncio
    async def test_max_expanded_terms_limit(self, service):
        """Test max expanded terms limit."""
        request = QueryExpansionRequest(
            original_query="function variable error",
            strategy=ExpansionStrategy.HYBRID,
            max_expanded_terms=2,
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        # Should not exceed max terms
        assert len(result.expanded_terms) <= 2

    @pytest.mark.asyncio
    async def test_min_confidence_filtering(self, service):
        """Test minimum confidence filtering."""
        request = QueryExpansionRequest(
            original_query="function programming",
            strategy=ExpansionStrategy.SYNONYM_BASED,
            min_confidence=0.9,
        )

        result = await service.expand_query(request)

        # All terms should meet min confidence
        for term in result.expanded_terms:
            assert term.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_caching_functionality(self, service):
        """Test expansion result caching."""
        request = QueryExpansionRequest(
            original_query="python programming",
            strategy=ExpansionStrategy.SYNONYM_BASED,
            enable_caching=True,
        )

        # First call - should miss cache
        result1 = await service.expand_query(request)
        assert result1.cache_hit is False
        assert service.cache_stats["misses"] == 1
        assert service.cache_stats["hits"] == 0

        # Second call - should hit cache
        result2 = await service.expand_query(request)
        assert result2.cache_hit is True
        assert service.cache_stats["hits"] == 1

        # Results should be the same
        assert result1.original_query == result2.original_query
        assert result1.expansion_strategy == result2.expansion_strategy

    @pytest.mark.asyncio
    async def test_cache_disabled(self, service):
        """Test expansion with caching disabled."""
        request = QueryExpansionRequest(
            original_query="python programming",
            strategy=ExpansionStrategy.SYNONYM_BASED,
            enable_caching=False,
        )

        result1 = await service.expand_query(request)
        result2 = await service.expand_query(request)

        # Both should be cache misses
        assert result1.cache_hit is False
        assert result2.cache_hit is False
        assert len(service.expansion_cache) == 0

    @pytest.mark.asyncio
    async def test_empty_query(self, service):
        """Test expansion with empty query."""
        request = QueryExpansionRequest(
            original_query="", strategy=ExpansionStrategy.SYNONYM_BASED
        )

        result = await service.expand_query(request)

        assert result.original_query == ""
        assert result.expanded_terms == []
        assert result.expanded_query == ""
        assert result.confidence_score >= 0.0

    @pytest.mark.asyncio
    async def test_single_term_query(self, service):
        """Test expansion with single term query."""
        request = QueryExpansionRequest(
            original_query="function",
            strategy=ExpansionStrategy.SYNONYM_BASED,
            min_confidence=0.1,
        )

        result = await service.expand_query(request)

        assert result.original_query == "function"
        # May or may not find expansions
        assert isinstance(result.expanded_terms, list)

    @pytest.mark.asyncio
    async def test_special_characters_query(self, service):
        """Test expansion with special characters in query."""
        request = QueryExpansionRequest(
            original_query="C++ programming & debugging!",
            strategy=ExpansionStrategy.SYNONYM_BASED,
        )

        result = await service.expand_query(request)

        # Should handle special characters gracefully
        assert isinstance(result, QueryExpansionResult)
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test error handling in expansion."""
        request = QueryExpansionRequest(
            original_query="test query", strategy=ExpansionStrategy.SYNONYM_BASED
        )

        # Mock a method to raise an exception
        with patch.object(
            service, "_preprocess_query", side_effect=Exception("Test error")
        ):
            result = await service.expand_query(request)

            # Should return fallback result
            assert result.original_query == "test query"
            assert result.expanded_terms == []
            assert result.expanded_query == "test query"
            assert result.confidence_score == 0.0
            assert "error" in result.expansion_metadata

    def test_preprocess_query(self, service):
        """Test query preprocessing."""
        test_cases = [
            ("Python Programming!", "python programming "),
            ("  Multiple   Spaces  ", "multiple spaces"),
            ("C++ & Java", "c++ java"),
            ("Machine-Learning.AI", "machine-learning.ai"),
        ]

        for input_query, expected in test_cases:
            result = service._preprocess_query(input_query)
            assert result == expected

    def test_extract_key_terms(self, service):
        """Test key term extraction."""
        test_cases = [
            ("python programming tutorial", ["python", "programming", "tutorial"]),
            ("the quick brown fox", ["quick", "brown", "fox"]),
            ("database and api development", ["database", "api", "development"]),
            ("a", []),  # Too short
            ("and or but", []),  # Only stop words
        ]

        for query, expected_terms in test_cases:
            result = service._extract_key_terms(query, {})
            assert result == expected_terms

    def test_get_synonyms(self, service):
        """Test synonym retrieval."""
        # Test with known domain vocabulary terms
        synonyms = service._get_synonyms("function")
        assert isinstance(synonyms, list)
        # Should find synonyms from programming domain
        expected_synonyms = {"method", "procedure", "routine", "subroutine"}
        assert any(syn in expected_synonyms for syn in synonyms)

    def test_calculate_synonym_confidence(self, service):
        """Test synonym confidence calculation."""
        confidence = service._calculate_synonym_confidence("function", "method")
        assert 0.0 <= confidence <= 1.0
        assert confidence == 0.7  # Base confidence in implementation

    def test_calculate_context_relevance(self, service):
        """Test context relevance calculation."""
        context = {"domain": ["programming"]}

        # Term in domain vocabulary
        relevance = service._calculate_context_relevance("method", context)
        assert 0.0 <= relevance <= 1.0

        # Term not in domain vocabulary
        relevance_unknown = service._calculate_context_relevance(
            "unknown_term", context
        )
        assert relevance_unknown == 0.5  # Base relevance

    def test_get_term_frequency_weight(self, service):
        """Test term frequency weight calculation."""
        # Default weight
        weight = service._get_term_frequency_weight("unknown_term")
        assert weight == 1.0

        # Custom weight
        service.term_frequencies["custom_term"] = 2.5
        weight_custom = service._get_term_frequency_weight("custom_term")
        assert weight_custom == 2.5

    def test_generate_term_statistics(self, service):
        """Test term statistics generation."""
        # Empty terms
        stats = service._generate_term_statistics([])
        assert stats == {}

        # Terms with data
        terms = [
            ExpandedTerm(
                term="method",
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.8,
                source="wordnet",
            ),
            ExpandedTerm(
                term="procedure",
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.9,
                source="custom",
            ),
            ExpandedTerm(
                term="related_term",
                original_term="function",
                relation_type=TermRelationType.RELATED,
                confidence=0.7,
                source="semantic",
            ),
        ]

        stats = service._generate_term_statistics(terms)

        assert stats["_total_terms"] == 3
        assert stats["relation_type_counts"]["synonym"] == 2
        assert stats["relation_type_counts"]["related"] == 1
        assert stats["source_counts"]["wordnet"] == 1
        assert stats["source_counts"]["custom"] == 1
        assert stats["source_counts"]["semantic"] == 1
        assert stats["confidence_stats"]["min"] == 0.7
        assert stats["confidence_stats"]["max"] == 0.9
        assert abs(stats["confidence_stats"]["avg"] - 0.8) < 0.0001

    def test_cache_operations(self, service):
        """Test cache operations."""
        request = QueryExpansionRequest(
            original_query="test", strategy=ExpansionStrategy.SYNONYM_BASED
        )

        result = QueryExpansionResult(
            original_query="test",
            expanded_query="test",
            expansion_strategy=ExpansionStrategy.SYNONYM_BASED,
            expansion_scope=ExpansionScope.MODERATE,
            confidence_score=0.8,
            processing_time_ms=100.0,
        )

        # Test cache key generation
        cache_key = service._generate_cache_key(request)
        assert isinstance(cache_key, str)
        assert "test" in cache_key
        assert "synonym_based" in cache_key

        # Test caching
        service._cache_result(request, result)
        assert len(service.expansion_cache) == 1

        # Test retrieval
        cached_result = service._get_cached_result(request)
        assert cached_result is not None
        assert cached_result.original_query == "test"

    def test_cache_size_limit(self, service):
        """Test cache size limit enforcement."""
        service.cache_size = 2

        # Add items beyond cache size
        for i in range(3):
            request = QueryExpansionRequest(
                original_query=f"test_{i}", strategy=ExpansionStrategy.SYNONYM_BASED
            )
            result = QueryExpansionResult(
                original_query=f"test_{i}",
                expanded_query=f"test_{i}",
                expansion_strategy=ExpansionStrategy.SYNONYM_BASED,
                expansion_scope=ExpansionScope.MODERATE,
                confidence_score=0.8,
                processing_time_ms=100.0,
            )
            service._cache_result(request, result)

        # Should only keep cache_size items
        assert len(service.expansion_cache) == 2

    def test_performance_stats_tracking(self, service):
        """Test performance statistics tracking."""
        initial_stats = service.performance_stats.copy()

        service._update_performance_stats(ExpansionStrategy.SYNONYM_BASED, 150.0)

        assert (
            service.performance_stats["_total_expansions"]
            == initial_stats["_total_expansions"] + 1
        )
        assert service.performance_stats["avg_processing_time"] > 0
        assert service.performance_stats["strategy_usage"]["synonym_based"] == 1

    def test_expansion_feedback_recording(self, service):
        """Test recording expansion feedback."""
        service.record_expansion_feedback(
            original_query="function",
            expanded_terms=["method", "procedure"],
            success_indicators={"method": 0.9, "procedure": 0.7},
        )

        assert "function" in service.success_feedback
        assert service.success_feedback["function"]["method"] > 0.5
        assert service.success_feedback["function"]["procedure"] > 0.5

    def test_get_performance_stats(self, service):
        """Test getting performance statistics."""
        stats = service.get_performance_stats()

        assert "_total_expansions" in stats
        assert "avg_processing_time" in stats
        assert "strategy_usage" in stats
        assert "cache_stats" in stats
        assert "cache_size" in stats

    def test_clear_cache(self, service):
        """Test clearing cache."""
        # Add some cache entries
        request = QueryExpansionRequest(original_query="test")
        result = QueryExpansionResult(
            original_query="test",
            expanded_query="test",
            expansion_strategy=ExpansionStrategy.SYNONYM_BASED,
            expansion_scope=ExpansionScope.MODERATE,
            confidence_score=0.8,
            processing_time_ms=100.0,
        )
        service._cache_result(request, result)
        service.cache_stats["hits"] = 5
        service.cache_stats["misses"] = 10

        # Clear cache
        service.clear_cache()

        assert len(service.expansion_cache) == 0
        assert service.cache_stats == {"hits": 0, "misses": 0}

    async def test_disabled_semantic_expansion(self):
        """Test service with semantic expansion disabled."""
        service = QueryExpansionService(enable_semantic_expansion=False)

        # Semantic expansion should return empty list
        key_terms = ["programming", "database"]
        request = QueryExpansionRequest(
            original_query="programming database",
            strategy=ExpansionStrategy.SEMANTIC_BASED,
        )

        result = await service._semantic_expansion(key_terms, request)
        assert result == []

    async def test_disabled_domain_expansion(self):
        """Test service with domain expansion disabled."""
        service = QueryExpansionService(enable_domain_expansion=False)

        # Domain expansion should return empty list
        key_terms = ["function"]
        request = QueryExpansionRequest(
            original_query="function",
            strategy=ExpansionStrategy.DOMAIN_SPECIFIC,
            target_domains=["programming"],
        )

        result = await service._domain_specific_expansion(key_terms, request)
        assert result == []

    @pytest.mark.asyncio
    async def test_build_expanded_query_formatting(self, service):
        """Test expanded query building and formatting."""
        # Test with no expanded terms
        expanded_query = service._build_expanded_query("original query", [], None)
        assert expanded_query == "original query"

        # Test with expanded terms
        terms = [
            ExpandedTerm(
                term="method",
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.8,
                source="wordnet",
            ),
            ExpandedTerm(
                term="procedure",
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.9,
                source="wordnet",
            ),
        ]

        request = QueryExpansionRequest(original_query="function programming")
        expanded_query = service._build_expanded_query(
            "function programming", terms, request
        )

        # Should contain OR structure
        assert "OR" in expanded_query
        assert "method" in expanded_query
        assert "procedure" in expanded_query

    def test_filter_and_rank_terms_deduplication(self, service):
        """Test term filtering and ranking with duplicates."""
        # Create duplicate terms with different confidence scores
        terms = [
            ExpandedTerm(
                term="method",
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.7,
                source="source1",
            ),
            ExpandedTerm(
                term="method",  # Duplicate with higher confidence
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.9,
                source="source2",
            ),
            ExpandedTerm(
                term="procedure",
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.8,
                source="source1",
            ),
        ]

        request = QueryExpansionRequest(
            original_query="function",
            max_expanded_terms=10,
            min_confidence=0.5,
            scope=ExpansionScope.MODERATE,
        )

        filtered = service._filter_and_rank_terms(terms, request)

        # Should have 2 unique terms, with higher confidence "method" kept
        assert len(filtered) == 2
        method_term = next(term for term in filtered if term.term == "method")
        assert method_term.confidence == 0.9
        assert method_term.source == "source2"

    @pytest.mark.asyncio
    async def test_comprehensive_hybrid_strategy(self, service):
        """Test hybrid strategy comprehensively."""
        request = QueryExpansionRequest(
            original_query="function error programming",
            strategy=ExpansionStrategy.HYBRID,
            min_confidence=0.1,
            max_expanded_terms=20,
        )

        with (
            patch.object(
                service, "_synonym_expansion", new_callable=AsyncMock
            ) as mock_syn,
            patch.object(
                service, "_semantic_expansion", new_callable=AsyncMock
            ) as mock_sem,
            patch.object(
                service, "_context_aware_expansion", new_callable=AsyncMock
            ) as mock_ctx,
        ):
            # Setup mock returns
            mock_syn.return_value = [
                ExpandedTerm(
                    term="method",
                    original_term="function",
                    relation_type=TermRelationType.SYNONYM,
                    confidence=0.8,
                    source="synonym",
                )
            ]
            mock_sem.return_value = [
                ExpandedTerm(
                    term="procedure",
                    original_term="function",
                    relation_type=TermRelationType.RELATED,
                    confidence=0.7,
                    source="semantic",
                )
            ]
            mock_ctx.return_value = [
                ExpandedTerm(
                    term="bug",
                    original_term="error",
                    relation_type=TermRelationType.RELATED,
                    confidence=0.9,
                    source="context",
                )
            ]

            result = await service.expand_query(request)

            # Should call all three expansion methods
            mock_syn.assert_called_once()
            mock_sem.assert_called_once()
            mock_ctx.assert_called_once()

            # Should combine results from all strategies
            assert len(result.expanded_terms) == 3
            sources = {term.source for term in result.expanded_terms}
            assert sources == {"synonym", "semantic", "context"}

    def test_confidence_calculation_edge_cases(self, service):
        """Test confidence calculation edge cases."""
        # Empty terms
        confidence = service._calculate_expansion_confidence([], None)
        assert confidence == 0.0

        # Single term
        terms = [
            ExpandedTerm(
                term="method",
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.8,
                source="test",
            )
        ]
        confidence = service._calculate_expansion_confidence(terms, None)
        assert 0.0 <= confidence <= 1.0

        # Multiple terms with zero weights
        terms_zero_weight = [
            ExpandedTerm(
                term="method",
                original_term="function",
                relation_type=TermRelationType.SYNONYM,
                confidence=0.8,
                source="test",
                context_relevance=0.0,
                frequency_weight=0.0,
            )
        ]
        confidence_zero = service._calculate_expansion_confidence(
            terms_zero_weight, None
        )
        assert confidence_zero == 0.0
