"""Query expansion service for synonym and related term discovery.

This module provides advanced query expansion capabilities including synonym generation,
related term discovery, semantic expansion, and intelligent query reformulation to
improve search recall and relevance.
"""

import logging
import re
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


class ExpansionStrategy(str, Enum):
    """Strategies for query expansion."""

    SYNONYM_BASED = "synonym_based"  # Classical synonym expansion
    SEMANTIC_BASED = "semantic_based"  # Vector similarity expansion
    CONTEXT_AWARE = "context_aware"  # Context-specific expansion
    DOMAIN_SPECIFIC = "domain_specific"  # Domain knowledge expansion
    HYBRID = "hybrid"  # Combined approach
    LEARNING_BASED = "learning_based"  # User behavior learning


class ExpansionScope(str, Enum):
    """Scope of query expansion."""

    CONSERVATIVE = "conservative"  # Add only highly confident terms
    MODERATE = "moderate"  # Balanced expansion
    AGGRESSIVE = "aggressive"  # Broad expansion for maximum recall
    ADAPTIVE = "adaptive"  # Adjust based on query context


class TermRelationType(str, Enum):
    """Types of term relationships."""

    SYNONYM = "synonym"  # Direct synonyms
    HYPERNYM = "hypernym"  # More general terms
    HYPONYM = "hyponym"  # More specific terms
    MERONYM = "meronym"  # Part-of relationships
    HOLONYM = "holonym"  # Whole-of relationships
    RELATED = "related"  # Semantically related
    ABBREVIATION = "abbreviation"  # Acronyms and abbreviations
    ALTERNATIVE = "alternative"  # Alternative spellings/forms


class ExpandedTerm(BaseModel):
    """An expanded term with metadata."""

    term: str = Field(..., description="The expanded term")
    original_term: str = Field(..., description="Original term that was expanded")
    relation_type: TermRelationType = Field(..., description="Type of relationship")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(
        ..., description="Source of expansion (model, dictionary, etc.)"
    )
    context_relevance: float = Field(
        1.0, ge=0.0, le=1.0, description="Relevance to query context"
    )
    frequency_weight: float = Field(
        1.0, ge=0.0, description="Frequency-based importance weight"
    )

    @field_validator("term", "original_term")
    @classmethod
    def validate_term_format(cls, v):
        """Validate term format."""
        if not v or not v.strip():
            msg = "Term cannot be empty"
            raise ValueError(msg)
        return v.strip().lower()


class QueryExpansionRequest(BaseModel):
    """Request for query expansion."""

    # Core query information
    original_query: str = Field(..., description="Original search query")
    query_context: dict[str, Any] = Field(
        default_factory=dict, description="Context information for the query"
    )

    # Expansion configuration
    strategy: ExpansionStrategy = Field(
        ExpansionStrategy.HYBRID, description="Expansion strategy to use"
    )
    scope: ExpansionScope = Field(
        ExpansionScope.MODERATE, description="Scope of expansion"
    )
    max_expanded_terms: int = Field(
        10, ge=1, le=50, description="Maximum number of expanded terms"
    )
    min_confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum confidence for expansion"
    )

    # Target domains and contexts
    target_domains: list[str] | None = Field(
        None, description="Specific domains to focus expansion on"
    )
    exclude_terms: list[str] | None = Field(
        None, description="Terms to exclude from expansion"
    )

    # Semantic constraints
    preserve_intent: bool = Field(
        True, description="Ensure expanded query preserves original intent"
    )
    boost_recent_terms: bool = Field(
        True, description="Boost terms that are currently relevant"
    )
    include_abbreviations: bool = Field(
        True, description="Include acronyms and abbreviations"
    )

    # Performance settings
    max_processing_time_ms: float = Field(
        2000.0, ge=100.0, description="Maximum processing time"
    )
    enable_caching: bool = Field(True, description="Enable expansion result caching")


class QueryExpansionResult(BaseModel):
    """Result of query expansion."""

    original_query: str = Field(..., description="Original query")
    expanded_terms: list[ExpandedTerm] = Field(
        default_factory=list, description="List of expanded terms"
    )
    expanded_query: str = Field(..., description="Final expanded query")

    # Expansion metadata
    expansion_strategy: ExpansionStrategy = Field(..., description="Strategy used")
    expansion_scope: ExpansionScope = Field(..., description="Scope used")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall expansion confidence"
    )

    # Performance metrics
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time")
    cache_hit: bool = Field(False, description="Whether result was cached")

    # Analysis metadata
    term_statistics: dict[str, Any] = Field(
        default_factory=dict, description="Statistics about expanded terms"
    )
    expansion_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional expansion metadata"
    )


class QueryExpansionService:
    """Advanced query expansion service with multiple strategies."""

    def __init__(
        self,
        enable_semantic_expansion: bool = True,
        enable_domain_expansion: bool = True,
        cache_size: int = 1000,
    ):
        """Initialize query expansion service.

        Args:
            enable_semantic_expansion: Enable semantic similarity expansion
            enable_domain_expansion: Enable domain-specific expansion
            cache_size: Size of expansion cache

        """
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Configuration
        self.enable_semantic_expansion = enable_semantic_expansion
        self.enable_domain_expansion = enable_domain_expansion

        # Caching
        self.expansion_cache = {}
        self.cache_size = cache_size
        self.cache_stats = {"hits": 0, "misses": 0}

        # Expansion sources and models
        self.synonym_sources = self._initialize_synonym_sources()
        self.domain_vocabularies = self._initialize_domain_vocabularies()
        self.term_frequencies = {}

        # Learning and adaptation
        self.expansion_history = []
        self.success_feedback = {}

        # Performance tracking
        self.performance_stats = {
            "total_expansions": 0,
            "avg_processing_time": 0.0,
            "strategy_usage": {},
        }

    async def expand_query(
        self, request: QueryExpansionRequest
    ) -> QueryExpansionResult:
        """Expand a query using the specified strategy.

        Args:
            request: Query expansion request

        Returns:
            QueryExpansionResult with expanded terms and metadata

        """
        start_time = time.time()

        try:
            # Check cache first
            if request.enable_caching:
                cached_result = self._get_cached_result(request)
                if cached_result:
                    cached_result.cache_hit = True
                    return cached_result

            # Preprocess query
            preprocessed_query = self._preprocess_query(request.original_query)

            # Extract key terms
            key_terms = self._extract_key_terms(
                preprocessed_query, request.query_context
            )

            # Generate expansions based on strategy
            expanded_terms = await self._generate_expansions(key_terms, request)

            # Filter and rank expanded terms
            filtered_terms = self._filter_and_rank_terms(expanded_terms, request)

            # Build expanded query
            expanded_query = self._build_expanded_query(
                preprocessed_query, filtered_terms, request
            )

            # Calculate confidence and statistics
            confidence_score = self._calculate_expansion_confidence(
                filtered_terms, request
            )

            processing_time_ms = (time.time() - start_time) * 1000

            # Build result
            result = QueryExpansionResult(
                original_query=request.original_query,
                expanded_terms=filtered_terms,
                expanded_query=expanded_query,
                expansion_strategy=request.strategy,
                expansion_scope=request.scope,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                cache_hit=False,
                term_statistics=self._generate_term_statistics(filtered_terms),
                expansion_metadata={
                    "key_terms_count": len(key_terms),
                    "expansion_sources": list({term.source for term in filtered_terms}),
                    "relation_types": list(
                        {term.relation_type for term in filtered_terms}
                    ),
                },
            )

            # Cache result
            if request.enable_caching:
                self._cache_result(request, result)

            # Update performance stats
            self._update_performance_stats(request.strategy, processing_time_ms)

            self._logger.info(
                f"Expanded query '{request.original_query}' to '{expanded_query}' "
                f"with {len(filtered_terms)} terms in {processing_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._logger.error(
                f"Query expansion failed: {e}", exc_info=True
            )  # TODO: Convert f-string to logging format

            # Return fallback result
            return QueryExpansionResult(
                original_query=request.original_query,
                expanded_terms=[],
                expanded_query=request.original_query,
                expansion_strategy=request.strategy,
                expansion_scope=request.scope,
                confidence_score=0.0,
                processing_time_ms=processing_time_ms,
                cache_hit=False,
                expansion_metadata={"error": str(e)},
            )

    async def _generate_expansions(
        self, key_terms: list[str], request: QueryExpansionRequest
    ) -> list[ExpandedTerm]:
        """Generate expansions using the specified strategy."""
        all_expanded_terms = []

        if request.strategy == ExpansionStrategy.SYNONYM_BASED:
            terms = await self._synonym_expansion(key_terms, request)
            all_expanded_terms.extend(terms)

        elif request.strategy == ExpansionStrategy.SEMANTIC_BASED:
            terms = await self._semantic_expansion(key_terms, request)
            all_expanded_terms.extend(terms)

        elif request.strategy == ExpansionStrategy.CONTEXT_AWARE:
            terms = await self._context_aware_expansion(key_terms, request)
            all_expanded_terms.extend(terms)

        elif request.strategy == ExpansionStrategy.DOMAIN_SPECIFIC:
            terms = await self._domain_specific_expansion(key_terms, request)
            all_expanded_terms.extend(terms)

        elif request.strategy == ExpansionStrategy.HYBRID:
            # Combine multiple strategies
            synonym_terms = await self._synonym_expansion(key_terms, request)
            semantic_terms = await self._semantic_expansion(key_terms, request)
            context_terms = await self._context_aware_expansion(key_terms, request)

            all_expanded_terms.extend(synonym_terms)
            all_expanded_terms.extend(semantic_terms)
            all_expanded_terms.extend(context_terms)

        elif request.strategy == ExpansionStrategy.LEARNING_BASED:
            terms = await self._learning_based_expansion(key_terms, request)
            all_expanded_terms.extend(terms)

        return all_expanded_terms

    async def _synonym_expansion(
        self, key_terms: list[str], request: QueryExpansionRequest
    ) -> list[ExpandedTerm]:
        """Generate synonym-based expansions."""
        expanded_terms = []

        for term in key_terms:
            # Get synonyms from various sources
            synonyms = self._get_synonyms(term)

            for synonym in synonyms:
                if synonym != term and synonym not in (request.exclude_terms or []):
                    confidence = self._calculate_synonym_confidence(term, synonym)

                    if confidence >= request.min_confidence:
                        expanded_terms.append(
                            ExpandedTerm(
                                term=synonym,
                                original_term=term,
                                relation_type=TermRelationType.SYNONYM,
                                confidence=confidence,
                                source="synonym_dictionary",
                                context_relevance=1.0,
                                frequency_weight=self._get_term_frequency_weight(
                                    synonym
                                ),
                            )
                        )

        return expanded_terms

    async def _semantic_expansion(
        self, key_terms: list[str], request: QueryExpansionRequest
    ) -> list[ExpandedTerm]:
        """Generate semantic similarity-based expansions."""
        if not self.enable_semantic_expansion:
            return []

        expanded_terms = []

        # This would typically use word embeddings or language models
        # For now, implementing a simplified version
        for term in key_terms:
            related_terms = self._get_semantically_related_terms(
                term, request.query_context
            )

            for related_term, similarity_score in related_terms:
                if related_term != term and related_term not in (
                    request.exclude_terms or []
                ):
                    confidence = (
                        similarity_score * 0.8
                    )  # Adjust for semantic uncertainty

                    if confidence >= request.min_confidence:
                        expanded_terms.append(
                            ExpandedTerm(
                                term=related_term,
                                original_term=term,
                                relation_type=TermRelationType.RELATED,
                                confidence=confidence,
                                source="semantic_model",
                                context_relevance=self._calculate_context_relevance(
                                    related_term, request.query_context
                                ),
                                frequency_weight=self._get_term_frequency_weight(
                                    related_term
                                ),
                            )
                        )

        return expanded_terms

    async def _context_aware_expansion(
        self, key_terms: list[str], request: QueryExpansionRequest
    ) -> list[ExpandedTerm]:
        """Generate context-aware expansions."""
        expanded_terms = []

        # Analyze query context
        context = request.query_context
        domain_hints = context.get("domain", [])
        user_intent = context.get("intent", "")

        for term in key_terms:
            # Get context-specific expansions
            context_terms = self._get_context_specific_terms(
                term, domain_hints, user_intent
            )

            for context_term in context_terms:
                if context_term != term and context_term not in (
                    request.exclude_terms or []
                ):
                    confidence = self._calculate_context_confidence(
                        term, context_term, context
                    )

                    if confidence >= request.min_confidence:
                        expanded_terms.append(
                            ExpandedTerm(
                                term=context_term,
                                original_term=term,
                                relation_type=TermRelationType.RELATED,
                                confidence=confidence,
                                source="context_analysis",
                                context_relevance=self._calculate_context_relevance(
                                    context_term, context
                                ),
                                frequency_weight=self._get_term_frequency_weight(
                                    context_term
                                ),
                            )
                        )

        return expanded_terms

    async def _domain_specific_expansion(
        self, key_terms: list[str], request: QueryExpansionRequest
    ) -> list[ExpandedTerm]:
        """Generate domain-specific expansions."""
        if not self.enable_domain_expansion:
            return []

        expanded_terms = []
        target_domains = request.target_domains or ["general"]

        for term in key_terms:
            for domain in target_domains:
                domain_terms = self._get_domain_specific_terms(term, domain)

                for domain_term, relevance_score in domain_terms:
                    if domain_term != term and domain_term not in (
                        request.exclude_terms or []
                    ):
                        confidence = relevance_score * 0.9  # Domain-specific confidence

                        if confidence >= request.min_confidence:
                            expanded_terms.append(
                                ExpandedTerm(
                                    term=domain_term,
                                    original_term=term,
                                    relation_type=TermRelationType.RELATED,
                                    confidence=confidence,
                                    source=f"domain_{domain}",
                                    context_relevance=relevance_score,
                                    frequency_weight=self._get_term_frequency_weight(
                                        domain_term
                                    ),
                                )
                            )

        return expanded_terms

    async def _learning_based_expansion(
        self, key_terms: list[str], request: QueryExpansionRequest
    ) -> list[ExpandedTerm]:
        """Generate expansions based on learned patterns."""
        expanded_terms = []

        # Use historical expansion success data
        for term in key_terms:
            learned_terms = self._get_learned_expansions(term, request.query_context)

            for learned_term, success_score in learned_terms:
                if learned_term != term and learned_term not in (
                    request.exclude_terms or []
                ):
                    confidence = success_score

                    if confidence >= request.min_confidence:
                        expanded_terms.append(
                            ExpandedTerm(
                                term=learned_term,
                                original_term=term,
                                relation_type=TermRelationType.RELATED,
                                confidence=confidence,
                                source="learning_model",
                                context_relevance=self._calculate_context_relevance(
                                    learned_term, request.query_context
                                ),
                                frequency_weight=self._get_term_frequency_weight(
                                    learned_term
                                ),
                            )
                        )

        return expanded_terms

    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query for expansion."""
        # Clean and normalize the query
        query = query.strip().lower()

        # Remove special characters but preserve important ones
        query = re.sub(r"[^\w\s\-\+\.]", " ", query)

        # Normalize whitespace
        return re.sub(r"\s+", " ", query)

    def _extract_key_terms(self, query: str, _context: dict[str, Any]) -> list[str]:
        """Extract key terms from the query for expansion."""
        # Simple term extraction - in production would use NLP
        words = query.split()

        # Filter out stop words and very short terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        key_terms = [word for word in words if len(word) > 2 and word not in stop_words]

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    def _filter_and_rank_terms(
        self, expanded_terms: list[ExpandedTerm], request: QueryExpansionRequest
    ) -> list[ExpandedTerm]:
        """Filter and rank expanded terms."""
        # Remove duplicates
        unique_terms = {}
        for term in expanded_terms:
            if (
                term.term not in unique_terms
                or term.confidence > unique_terms[term.term].confidence
            ):
                unique_terms[term.term] = term

        filtered_terms = list(unique_terms.values())

        # Apply confidence filter
        filtered_terms = [
            term for term in filtered_terms if term.confidence >= request.min_confidence
        ]

        # Apply scope-based filtering
        if request.scope == ExpansionScope.CONSERVATIVE:
            filtered_terms = [term for term in filtered_terms if term.confidence >= 0.8]
        elif request.scope == ExpansionScope.MODERATE:
            filtered_terms = [term for term in filtered_terms if term.confidence >= 0.6]
        # AGGRESSIVE and ADAPTIVE keep all terms above min_confidence

        # Rank by composite score
        def calculate_composite_score(term: ExpandedTerm) -> float:
            return (
                term.confidence * 0.4
                + term.context_relevance * 0.3
                + min(1.0, term.frequency_weight) * 0.3
            )

        filtered_terms.sort(key=calculate_composite_score, reverse=True)

        # Limit to max terms
        return filtered_terms[: request.max_expanded_terms]

    def _build_expanded_query(
        self,
        original_query: str,
        expanded_terms: list[ExpandedTerm],
        _request: QueryExpansionRequest,
    ) -> str:
        """Build the final expanded query."""
        if not expanded_terms:
            return original_query

        # Group terms by original term
        term_groups = {}
        for expanded_term in expanded_terms:
            if expanded_term.original_term not in term_groups:
                term_groups[expanded_term.original_term] = []
            term_groups[expanded_term.original_term].append(expanded_term.term)

        # Build expanded query
        expanded_query = original_query

        for original_term, related_terms in term_groups.items():
            if original_term in expanded_query:
                # Add related terms as OR alternatives
                all_terms = [original_term] + related_terms[
                    :3
                ]  # Limit to avoid overly complex queries
                term_expansion = f"({' OR '.join(all_terms)})"
                expanded_query = expanded_query.replace(
                    original_term, term_expansion, 1
                )

        return expanded_query

    def _calculate_expansion_confidence(
        self, expanded_terms: list[ExpandedTerm], _request: QueryExpansionRequest
    ) -> float:
        """Calculate overall expansion confidence."""
        if not expanded_terms:
            return 0.0

        # Weighted average of term confidences
        total_weight = 0.0
        weighted_confidence = 0.0

        for term in expanded_terms:
            weight = term.context_relevance * term.frequency_weight
            weighted_confidence += term.confidence * weight
            total_weight += weight

        return weighted_confidence / total_weight if total_weight > 0 else 0.0

    def _generate_term_statistics(
        self, expanded_terms: list[ExpandedTerm]
    ) -> dict[str, Any]:
        """Generate statistics about expanded terms."""
        if not expanded_terms:
            return {}

        relation_counts = {}
        source_counts = {}
        confidence_stats = []

        for term in expanded_terms:
            # Count relation types
            rel_type = term.relation_type.value
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1

            # Count sources
            source_counts[term.source] = source_counts.get(term.source, 0) + 1

            # Collect confidence scores
            confidence_stats.append(term.confidence)

        return {
            "total_terms": len(expanded_terms),
            "relation_type_counts": relation_counts,
            "source_counts": source_counts,
            "confidence_stats": {
                "min": min(confidence_stats),
                "max": max(confidence_stats),
                "avg": sum(confidence_stats) / len(confidence_stats),
                "median": sorted(confidence_stats)[len(confidence_stats) // 2],
            },
        }

    def _initialize_synonym_sources(self) -> dict[str, Any]:
        """Initialize synonym dictionaries and sources."""
        # In production, this would load actual synonym databases
        return {
            "wordnet": {},  # WordNet synonyms
            "custom": {},  # Custom domain synonyms
            "learned": {},  # Machine learned synonyms
        }

    def _initialize_domain_vocabularies(self) -> dict[str, dict[str, list[str]]]:
        """Initialize domain-specific vocabularies."""
        # Sample domain vocabularies - in production load from external sources
        return {
            "programming": {
                "function": ["method", "procedure", "routine", "subroutine"],
                "variable": ["var", "parameter", "argument", "field"],
                "error": ["exception", "bug", "fault", "issue"],
            },
            "general": {
                "fast": ["quick", "rapid", "swift", "speedy"],
                "good": ["excellent", "great", "fine", "quality"],
                "big": ["large", "huge", "massive", "enormous"],
            },
        }

    def _get_synonyms(self, term: str) -> list[str]:
        """Get synonyms for a term."""
        synonyms = []

        # Check custom domain vocabularies
        for vocab in self.domain_vocabularies.values():
            if term in vocab:
                synonyms.extend(vocab[term])

        # In production, would also check external APIs like WordNet, etc.

        return list(set(synonyms))  # Remove duplicates

    def _get_semantically_related_terms(
        self, term: str, _context: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Get semantically related terms using embeddings."""
        # Simplified implementation - in production would use actual embeddings
        related_terms = []

        # Sample related terms with similarity scores
        sample_relations = {
            "programming": [("coding", 0.9), ("development", 0.8), ("software", 0.7)],
            "search": [("find", 0.8), ("query", 0.7), ("lookup", 0.6)],
            "database": [("storage", 0.7), ("repository", 0.6), ("data", 0.8)],
        }

        if term in sample_relations:
            related_terms.extend(sample_relations[term])

        return related_terms

    def _get_context_specific_terms(
        self, term: str, domain_hints: list[str], _user_intent: str
    ) -> list[str]:
        """Get context-specific term expansions."""
        context_terms = []

        # Simple context-based expansion
        if "programming" in domain_hints:
            programming_context = {
                "function": ["method", "procedure"],
                "error": ["exception", "bug"],
                "data": ["information", "dataset"],
            }
            if term in programming_context:
                context_terms.extend(programming_context[term])

        return context_terms

    def _get_domain_specific_terms(
        self, term: str, domain: str
    ) -> list[tuple[str, float]]:
        """Get domain-specific related terms."""
        domain_terms = []

        if (
            domain in self.domain_vocabularies
            and term in self.domain_vocabularies[domain]
        ):
            domain_terms.extend(
                [
                    (related_term, 0.8)  # High relevance for domain terms
                    for related_term in self.domain_vocabularies[domain][term]
                ]
            )

        return domain_terms

    def _get_learned_expansions(
        self, term: str, _context: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Get expansions based on learned patterns."""
        # Simplified implementation - would use actual ML model
        learned_terms = []

        # Use success feedback from previous expansions
        if term in self.success_feedback:
            for expansion, success_rate in self.success_feedback[term].items():
                learned_terms.append((expansion, success_rate))

        return learned_terms

    def _calculate_synonym_confidence(self, _original: str, _synonym: str) -> float:
        """Calculate confidence for synonym relationships."""
        # Simple confidence calculation - in production would be more sophisticated
        return 0.7

        # Boost confidence for exact matches in dictionaries
        # Reduce confidence for approximate matches

    def _calculate_context_relevance(self, term: str, context: dict[str, Any]) -> float:
        """Calculate relevance of term to query context."""
        # Simplified relevance calculation
        relevance = 0.5  # Base relevance

        domain_hints = context.get("domain", [])
        for domain in domain_hints:
            if domain in self.domain_vocabularies:
                for term_list in self.domain_vocabularies[domain].values():
                    if term in term_list:
                        relevance = min(1.0, relevance + 0.3)

        return relevance

    def _calculate_context_confidence(
        self, _original: str, context_term: str, context: dict[str, Any]
    ) -> float:
        """Calculate confidence for context-based expansion."""
        base_confidence = 0.6

        # Adjust based on context strength
        context_relevance = self._calculate_context_relevance(context_term, context)

        return min(1.0, base_confidence + context_relevance * 0.3)

    def _get_term_frequency_weight(self, term: str) -> float:
        """Get frequency-based weight for a term."""
        # Simple frequency weighting - in production would use actual frequency data
        return self.term_frequencies.get(term, 1.0)

    def _get_cached_result(
        self, request: QueryExpansionRequest
    ) -> QueryExpansionResult | None:
        """Get cached expansion result."""
        cache_key = self._generate_cache_key(request)

        if cache_key in self.expansion_cache:
            self.cache_stats["hits"] += 1
            return self.expansion_cache[cache_key]

        self.cache_stats["misses"] += 1
        return None

    def _cache_result(
        self, request: QueryExpansionRequest, result: QueryExpansionResult
    ) -> None:
        """Cache expansion result."""
        if len(self.expansion_cache) >= self.cache_size:
            # Simple LRU eviction - remove oldest entry
            oldest_key = next(iter(self.expansion_cache))
            del self.expansion_cache[oldest_key]

        cache_key = self._generate_cache_key(request)
        self.expansion_cache[cache_key] = result

    def _generate_cache_key(self, request: QueryExpansionRequest) -> str:
        """Generate cache key for request."""
        # Simple cache key based on query and strategy
        key_components = [
            request.original_query,
            request.strategy.value,
            request.scope.value,
            str(request.max_expanded_terms),
            str(request.min_confidence),
        ]
        return "|".join(key_components)

    def _update_performance_stats(
        self, strategy: ExpansionStrategy, processing_time: float
    ) -> None:
        """Update performance statistics."""
        self.performance_stats["total_expansions"] += 1

        # Update average processing time
        total = self.performance_stats["total_expansions"]
        current_avg = self.performance_stats["avg_processing_time"]
        self.performance_stats["avg_processing_time"] = (
            current_avg * (total - 1) + processing_time
        ) / total

        # Update strategy usage
        strategy_key = strategy.value
        if strategy_key not in self.performance_stats["strategy_usage"]:
            self.performance_stats["strategy_usage"][strategy_key] = 0
        self.performance_stats["strategy_usage"][strategy_key] += 1

    def record_expansion_feedback(
        self,
        original_query: str,
        expanded_terms: list[str],
        success_indicators: dict[str, float],
    ) -> None:
        """Record feedback on expansion success for learning."""
        for term in expanded_terms:
            if original_query not in self.success_feedback:
                self.success_feedback[original_query] = {}

            # Update success rate based on feedback
            current_rate = self.success_feedback[original_query].get(term, 0.5)
            feedback_score = success_indicators.get(term, 0.5)

            # Simple learning rate
            learning_rate = 0.1
            new_rate = current_rate + learning_rate * (feedback_score - current_rate)

            self.success_feedback[original_query][term] = new_rate

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.performance_stats,
            "cache_stats": self.cache_stats,
            "cache_size": len(self.expansion_cache),
        }

    def clear_cache(self) -> None:
        """Clear expansion cache."""
        self.expansion_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
