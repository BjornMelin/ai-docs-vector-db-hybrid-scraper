"""Query Preprocessing and Enhancement Pipeline.

This module provides comprehensive query preprocessing including normalization,
spell correction, expansion, and context extraction for optimal query processing.
"""

import logging
import re
import string
import time
from typing import Any

from .models import QueryPreprocessingResult


logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Advanced query preprocessing for enhanced search quality.

    Provides spell correction, normalization, expansion, and context extraction
    to prepare queries for optimal processing by the query pipeline.
    """

    def __init__(self):
        """Initialize the query preprocessor."""
        self._initialized = False

        # Common spelling corrections for technical terms
        self._spelling_corrections = {
            "javascipt": "javascript",
            "phython": "python",
            "databse": "database",
            "microservice": "microservices",
            "authetication": "authentication",
            "authoriztion": "authorization",
            "optimisation": "optimization",
            "performace": "performance",
            "intergration": "integration",
            "confguration": "configuration",
            "achitecture": "architecture",
            "secuirty": "security",
        }

        # Synonym expansions for better search coverage
        self._synonym_expansions = {
            "api": ["rest api", "web api", "service api"],
            "db": ["database"],
            "auth": ["authentication", "authorization"],
            "config": ["configuration", "setup"],
            "perf": ["performance"],
            "sec": ["security"],
            "js": ["javascript"],
            "py": ["python"],
            "ml": ["machine learning"],
            "ai": ["artificial intelligence"],
            "ci/cd": ["continuous integration", "continuous deployment"],
            "k8s": ["kubernetes"],
            "docker": ["containerization"],
        }

        # Stop words that can be removed for better search
        self._stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
        }

        # Technical context patterns
        self._context_patterns = {
            "programming_language": [
                r"\b(?:python|javascript|java|cpp|csharp|go|rust|php|ruby)\b",
                r"\b(?:js|py|c\+\+|c#)\b",
            ],
            "framework": [
                r"\b(?:react|vue|angular|django|flask|spring|express|laravel)\b",
                r"\b(?:rails|nextjs|nuxt|svelte)\b",
            ],
            "database": [
                r"\b(?:mysql|postgresql|mongodb|redis|sqlite|oracle)\b",
                r"\b(?:sql|nosql|database|db)\b",
            ],
            "cloud_platform": [
                r"\b(?:aws|azure|gcp|google cloud|amazon web services)\b",
                r"\b(?:ec2|s3|lambda|kubernetes|docker)\b",
            ],
            "version": [
                r"\bversion\s+(\d+(?:\.\d+)*)\b",
                r"\bv(\d+(?:\.\d+)*)\b",
                r"\b(\d+(?:\.\d+)+)\b",
            ],
            "error_code": [
                r"\b(?:error|exception)\s+(\w+\d+|\d+)\b",
                r"\b(\d{3,4})\s+(?:error|status)\b",
            ],
        }

    async def initialize(self) -> None:
        """Initialize the preprocessor."""
        self._initialized = True
        logger.info("QueryPreprocessor initialized")

    async def preprocess_query(
        self,
        query: str,
        enable_spell_correction: bool = True,
        enable_expansion: bool = True,
        enable_normalization: bool = True,
        enable_context_extraction: bool = True,
    ) -> QueryPreprocessingResult:
        """Preprocess query with comprehensive enhancements.

        Args:
            query: Original query to preprocess
            enable_spell_correction: Whether to apply spell corrections
            enable_expansion: Whether to add synonym expansions
            enable_normalization: Whether to normalize text
            enable_context_extraction: Whether to extract context information

        Returns:
            QueryPreprocessingResult: Preprocessing results with enhanced query

        Raises:
            RuntimeError: If preprocessor not initialized

        """
        if not self._initialized:
            msg = "QueryPreprocessor not initialized"
            raise RuntimeError(msg)

        start_time = time.time()

        # Track what preprocessing steps were applied
        corrections_applied = []
        expansions_added = []
        normalization_applied = False
        context_extracted = {}

        # Start with original query
        processed_query = query.strip()

        # 1. Extract context before modifying query
        if enable_context_extraction:
            context_extracted = self._extract_context(processed_query)

        # 2. Apply spell corrections
        if enable_spell_correction:
            corrected_query, corrections = self._apply_spell_corrections(
                processed_query
            )
            if corrections:
                processed_query = corrected_query
                corrections_applied.extend(corrections)

        # 3. Normalize text
        if enable_normalization:
            normalized_query = self._normalize_text(processed_query)
            if normalized_query != processed_query:
                processed_query = normalized_query
                normalization_applied = True

        # 4. Add synonym expansions (carefully to avoid query bloat)
        if enable_expansion:
            expanded_query, expansions = self._add_expansions(processed_query)
            if expansions:
                processed_query = expanded_query
                expansions_added.extend(expansions)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        return QueryPreprocessingResult(
            original_query=query,
            processed_query=processed_query,
            corrections_applied=corrections_applied,
            expansions_added=expansions_added,
            normalization_applied=normalization_applied,
            context_extracted=context_extracted,
            preprocessing_time_ms=processing_time_ms,
        )

    def _apply_spell_corrections(self, query: str) -> tuple[str, list[str]]:
        """Apply spell corrections to technical terms."""
        corrected_query = query
        corrections_applied = []

        for misspelled, correct in self._spelling_corrections.items():
            if misspelled in query.lower():
                # Case-sensitive replacement
                pattern = re.compile(re.escape(misspelled), re.IGNORECASE)
                if pattern.search(corrected_query):
                    corrected_query = pattern.sub(correct, corrected_query)
                    corrections_applied.append(f"{misspelled} → {correct}")

        return corrected_query, corrections_applied

    def _normalize_text(self, query: str) -> str:
        """Normalize query text for consistent processing."""
        # Convert to lowercase for processing but preserve original case for display
        normalized = query.strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Normalize quotation marks
        normalized = re.sub(r'[""' ",„]", '"', normalized)

        # Normalize hyphens and dashes
        normalized = re.sub(r"[-—]", "-", normalized)

        # Remove redundant punctuation (but keep meaningful punctuation)
        normalized = re.sub(r"([.!?]){2,}", r"\1", normalized)

        # Normalize common abbreviations
        abbreviation_map = {
            r"\bw/\b": "with",
            r"\bw/o\b": "without",
            r"\be\.g\.\b": "for example",
            r"\bi\.e\.\b": "that is",
            r"\betc\.?\b": "and so on",
        }

        for pattern, replacement in abbreviation_map.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        return normalized.strip()

    def _add_expansions(self, query: str) -> tuple[str, list[str]]:
        """Add relevant synonym expansions without making query too long."""
        query_lower = query.lower()
        expansions_added = []
        expanded_parts = []

        # Split query into words for analysis
        words = query_lower.split()

        # Only add expansions if query is short enough
        if len(words) <= 8:  # Avoid expanding very long queries
            for abbreviation, expansions in self._synonym_expansions.items():
                if abbreviation in query_lower and expansions:
                    best_expansion = expansions[0]
                    if best_expansion not in query_lower:
                        expanded_parts.append(best_expansion)
                        expansions_added.append(f"{abbreviation} → {best_expansion}")

        # Combine original query with expansions
        if expanded_parts:
            # Add expansions in parentheses to maintain query clarity
            expansion_text = " (" + " ".join(expanded_parts) + ")"
            return query + expansion_text, expansions_added

        return query, expansions_added

    def _extract_context(self, query: str) -> dict[str, Any]:
        """Extract contextual information from the query."""
        context = {}

        for context_type, patterns in self._context_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, query, re.IGNORECASE)
                if found:
                    matches.extend(found)

            if matches:
                # Remove duplicates while preserving order
                unique_matches = list(dict.fromkeys(matches))
                context[context_type] = unique_matches

        # Extract technical complexity indicators
        complexity_indicators = []
        if re.search(
            r"\b(?:architecture|design pattern|scalability)\b", query, re.IGNORECASE
        ):
            complexity_indicators.append("architectural")
        if re.search(
            r"\b(?:performance|optimization|latency|throughput)\b", query, re.IGNORECASE
        ):
            complexity_indicators.append("performance")
        if re.search(
            r"\b(?:security|authentication|encryption)\b", query, re.IGNORECASE
        ):
            complexity_indicators.append("security")
        if re.search(
            r"\b(?:integration|api|webhook|microservice)\b", query, re.IGNORECASE
        ):
            complexity_indicators.append("integration")

        if complexity_indicators:
            context["complexity_indicators"] = complexity_indicators

        # Extract urgency/priority indicators
        urgency_indicators = []
        if re.search(
            r"\b(?:urgent|asap|critical|emergency|production)\b", query, re.IGNORECASE
        ):
            urgency_indicators.append("high")
        elif re.search(r"\b(?:soon|quick|fast|immediate)\b", query, re.IGNORECASE):
            urgency_indicators.append("medium")

        if urgency_indicators:
            context["urgency"] = urgency_indicators[0]

        # Extract experience level indicators
        experience_indicators = []
        if re.search(
            r"\b(?:beginner|new to|just started|learning)\b", query, re.IGNORECASE
        ):
            experience_indicators.append("beginner")
        elif re.search(
            r"\b(?:advanced|expert|professional|enterprise)\b", query, re.IGNORECASE
        ):
            experience_indicators.append("advanced")
        elif re.search(r"\b(?:intermediate|some experience)\b", query, re.IGNORECASE):
            experience_indicators.append("intermediate")

        if experience_indicators:
            context["experience_level"] = experience_indicators[0]

        return context

    def _remove_stop_words(self, query: str) -> str:
        """Remove stop words while preserving meaning."""
        words = query.split()

        # Only remove stop words if query is long enough
        if len(words) <= 4:
            return query

        # Preserve important stop words that affect meaning
        important_stop_words = {
            "not",
            "no",
            "without",
            "from",
            "to",
            "in",
            "on",
            "with",
        }

        filtered_words = []
        for word in words:
            word_clean = word.lower().strip(string.punctuation)
            if word_clean not in self._stop_words or word_clean in important_stop_words:
                filtered_words.append(word)

        return " ".join(filtered_words) if filtered_words else query

    async def cleanup(self) -> None:
        """Cleanup preprocessor resources."""
        self._initialized = False
        logger.info("QueryPreprocessor cleaned up")
