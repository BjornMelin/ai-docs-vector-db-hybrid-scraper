"""Query classifier for adaptive hybrid search optimization.

This module implements ML-based query classification to determine optimal
search strategies based on query characteristics and complexity.
"""

import logging
import re
from typing import Any

from src.config import Config, QueryComplexity, QueryType
from src.models.vector_search import QueryClassification, QueryFeatures


logger = logging.getLogger(__name__)


class QueryClassifier:
    """ML-based query classifier for adaptive search optimization."""

    def __init__(self, config: Config):
        """Initialize query classifier.

        Args:
            config: Unified configuration

        """
        self.config = config
        self._programming_keywords = {
            "function",
            "method",
            "class",
            "variable",
            "array",
            "object",
            "string",
            "integer",
            "boolean",
            "import",
            "export",
            "async",
            "await",
            "promise",
            "callback",
            "interface",
            "type",
            "enum",
            "generic",
            "template",
            "module",
            "package",
            "library",
            "framework",
            "api",
            "endpoint",
            "database",
            "query",
            "schema",
            "model",
            "controller",
            "service",
            "component",
            "directive",
            "decorator",
            "annotation",
            "inheritance",
            "polymorphism",
            "encapsulation",
            "abstraction",
            "algorithm",
            "data structure",
            "recursion",
            "iteration",
            "loop",
            "condition",
            "exception",
            "error",
            "debug",
            "test",
            "unit test",
            "integration",
            "deployment",
            "build",
            "compile",
            "runtime",
            "memory",
            "performance",
            "optimization",
            "refactor",
            "version control",
            "git",
            "repository",
        }

        self._programming_languages = {
            "python",
            "javascript",
            "typescript",
            "java",
            "c++",
            "c#",
            "go",
            "rust",
            "swift",
            "kotlin",
            "scala",
            "ruby",
            "php",
            "html",
            "css",
            "sql",
            "bash",
            "shell",
            "powershell",
            "react",
            "vue",
            "angular",
            "node",
            "django",
            "flask",
            "spring",
            "laravel",
            "rails",
            "express",
            ".net",
            "pandas",
            "numpy",
            "tensorflow",
            "pytorch",
            "sklearn",
        }

        self._question_patterns = {
            "how": r"\bhow\s+(to|do|can|should)\b",
            "what": r"\bwhat\s+(is|are|does|means?)\b",
            "why": r"\bwhy\s+(is|are|does|do)\b",
            "when": r"\bwhen\s+(to|should|do)\b",
            "where": r"\bwhere\s+(is|are|can|to)\b",
            "which": r"\bwhich\s+(is|are|one|way)\b",
            "implement": r"\b(implement|create|build|make|develop)\b",
            "debug": r"\b(debug|fix|solve|error|issue|problem)\b",
            "compare": r"\b(compare|difference|vs|versus|better)\b",
            "tutorial": r"\b(tutorial|guide|learn|example|demo)\b",
        }

        self._code_syntax_patterns = [
            r"\w+\.\w+\(",  # Method calls
            r"def\s+\w+\(",  # Python functions
            r"function\s+\w+\(",  # JavaScript functions
            r"class\s+\w+",  # Class definitions
            r"import\s+\w+",  # Import statements
            r"from\s+\w+\s+import",  # Python imports
            r"#include\s*<",  # C++ includes
            r"@\w+",  # Decorators/annotations
            r"\{\s*\w+:\s*\w+\s*\}",  # Object literals
            r"\[\s*\w+\s*for\s*\w+\s*in\s*\w+\s*\]",  # List comprehensions
        ]

    async def classify_query(
        self, query: str, _context: dict[str, Any] | None = None
    ) -> QueryClassification:
        """Classify a query to determine optimal search strategy.

        Args:
            query: The search query text
            context: Optional context information (user history, session data)

        Returns:
            QueryClassification:
                Classification result with type, complexity, and features

        """
        try:
            # Extract features from query
            features = self._extract_features(query)

            # Determine query type
            query_type = self._classify_query_type(query, features)

            # Assess complexity
            complexity = self._assess_complexity(query, features)

            # Detect domain and programming language
            domain = self._detect_domain(query, features)
            programming_language = self._detect_programming_language(query, features)

            # Calculate overall confidence
            confidence = self._calculate_confidence(
                query, features, query_type, complexity
            )

            # Check for multimodal indicators
            is_multimodal = self._detect_multimodal(query, features)

            # Create and return proper QueryClassification object
            return QueryClassification(
                query_type=query_type.value,
                complexity_level=complexity.value,
                domain=domain,
                programming_language=programming_language,
                is_multimodal=is_multimodal,
                confidence=confidence,
                features=features.model_dump()
                if hasattr(features, "model_dump")
                else features.__dict__,
            )

        except Exception:
            logger.exception("Query classification failed")
            # Return default classification as QueryClassification object
            return QueryClassification(
                query_type=QueryType.CONCEPTUAL.value,
                complexity_level=QueryComplexity.MODERATE.value,
                domain="general",
                programming_language=None,
                is_multimodal=False,
                confidence=0.5,
                features={},
            )

    def _extract_features(self, query: str) -> QueryFeatures:
        """Extract features from query for classification."""
        query_lower = query.lower()
        tokens = query_lower.split()

        # Basic features
        query_length = len(tokens)

        # Programming-related features
        has_code_keywords = any(
            keyword in query_lower for keyword in self._programming_keywords
        )
        has_function_names = bool(re.search(r"\w+\(\)", query))
        has_programming_syntax = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self._code_syntax_patterns
        )

        # Question analysis
        question_type = self._identify_question_type(query_lower)

        # Technical depth assessment
        technical_depth = self._assess_technical_depth(query_lower, tokens)

        # Entity extraction (simple approach)
        entity_mentions = self._extract_entities(query_lower)

        # Programming language indicators
        programming_language_indicators = [
            lang for lang in self._programming_languages if lang in query_lower
        ]

        # Semantic complexity scoring
        semantic_complexity = self._calculate_semantic_complexity(query_lower, tokens)

        # Keyword density
        keyword_density = self._calculate_keyword_density(query_lower, tokens)

        return QueryFeatures(
            query_length=query_length,
            has_code_keywords=has_code_keywords,
            has_function_names=has_function_names,
            has_programming_syntax=has_programming_syntax,
            question_type=question_type or "",
            technical_depth=technical_depth,
            entity_mentions=entity_mentions,
            programming_language_indicators=programming_language_indicators,
            semantic_complexity=semantic_complexity,
            keyword_density=keyword_density,
        )

    def _classify_query_type(self, query: str, features: QueryFeatures) -> QueryType:
        """Classify the primary query type."""
        query_lower = query.lower()

        # Code search indicators
        if (
            features.has_programming_syntax
            or features.has_function_names
            or len(features.programming_language_indicators) > 0
        ):
            return QueryType.CODE

        # API reference indicators
        if any(
            term in query_lower
            for term in ["api", "reference", "documentation", "docs"]
        ) and any(
            term in query_lower
            for term in ["endpoint", "method", "parameter", "response"]
        ):
            return QueryType.API_REFERENCE

        # Documentation indicators
        if any(
            term in query_lower
            for term in ["documentation", "docs", "guide", "manual", "specification"]
        ):
            return QueryType.DOCUMENTATION

        # Troubleshooting indicators
        if any(
            term in query_lower
            for term in [
                "error",
                "issue",
                "problem",
                "bug",
                "fix",
                "debug",
                "troubleshoot",
                "not working",
                "fails",
                "broken",
            ]
        ):
            return QueryType.TROUBLESHOOTING

        # Multimodal indicators
        if features.has_code_keywords and any(
            term in query_lower
            for term in ["example", "tutorial", "demo", "screenshot", "image", "video"]
        ):
            return QueryType.MULTIMODAL

        # Default to conceptual
        return QueryType.CONCEPTUAL

    def _assess_complexity(
        self,
        query: str,
        features: QueryFeatures,
    ) -> QueryComplexity:
        """Assess query complexity level."""
        complexity_score = 0

        # Length-based complexity
        if features.query_length > 10:
            complexity_score += 1
        if features.query_length > 20:
            complexity_score += 1

        # Technical complexity
        if features.has_programming_syntax:
            complexity_score += 2
        if features.keyword_density > 0.5:
            complexity_score += 1

        # Question complexity
        if features.question_type in ["how", "implement"]:
            complexity_score += 1
        if features.question_type in ["compare", "why"]:
            complexity_score += 2

        # Multi-hop indicators
        multi_hop_keywords = [
            "and",
            "then",
            "after",
            "also",
            "additionally",
            "furthermore",
        ]
        if any(keyword in query.lower() for keyword in multi_hop_keywords):
            complexity_score += 1

        # Classify based on score
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        if complexity_score <= 4:
            return QueryComplexity.MODERATE
        return QueryComplexity.COMPLEX

    def _detect_domain(self, query: str, features: QueryFeatures) -> str:
        """Detect the technical domain of the query."""
        query_lower = query.lower()

        # Programming domains
        if features.programming_language_indicators:
            return "programming"

        # Web development
        if any(
            term in query_lower
            for term in [
                "web",
                "frontend",
                "backend",
                "html",
                "css",
                "javascript",
                "react",
                "vue",
            ]
        ):
            return "web_development"

        # Data science
        if any(
            term in query_lower
            for term in [
                "data",
                "analysis",
                "machine learning",
                "ai",
                "pandas",
                "numpy",
                "sklearn",
            ]
        ):
            return "data_science"

        # DevOps
        if any(
            term in query_lower
            for term in ["docker", "kubernetes", "deployment", "ci/cd", "aws", "cloud"]
        ):
            return "devops"

        # Mobile development
        if any(
            term in query_lower
            for term in ["mobile", "android", "ios", "swift", "kotlin", "react native"]
        ):
            return "mobile_development"

        return "general"

    def _detect_programming_language(
        self,
        _query: str,
        features: QueryFeatures,
    ) -> str | None:
        """Detect the primary programming language mentioned in the query."""
        if features.programming_language_indicators:
            # Return the first detected language (could be enhanced with ranking)
            return features.programming_language_indicators[0]
        return None

    def _calculate_confidence(
        self,
        _query: str,
        features: QueryFeatures,
        _query_type: QueryType,
        _complexity: QueryComplexity,
    ) -> float:
        """Calculate confidence score for the classification."""
        confidence = 0.5  # Base confidence

        # Increase confidence based on clear indicators
        if features.has_programming_syntax:
            confidence += 0.2
        if features.programming_language_indicators:
            confidence += 0.15
        if features.has_code_keywords:
            confidence += 0.1
        if features.question_type:
            confidence += 0.1

        # Adjust based on query length (very short/long queries are less reliable)
        if 5 <= features.query_length <= 15:
            confidence += 0.05
        elif features.query_length < 3 or features.query_length > 25:
            confidence -= 0.1

        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0

    def _detect_multimodal(
        self, query: str, _features: Any
    ) -> bool:  # TODO: Replace with proper QueryFeatures type
        """Detect if query involves multiple modalities."""
        query_lower = query.lower()
        multimodal_keywords = [
            "image",
            "picture",
            "screenshot",
            "diagram",
            "chart",
            "graph",
            "video",
            "audio",
            "visual",
            "example",
            "demo",
            "illustration",
        ]
        return any(keyword in query_lower for keyword in multimodal_keywords)

    def _identify_question_type(self, query_lower: str) -> str | None:
        """Identify the type of question being asked."""
        for question_type, pattern in self._question_patterns.items():
            if re.search(pattern, query_lower):
                return question_type
        return None

    def _assess_technical_depth(self, query_lower: str, _tokens: list[str]) -> str:
        """Assess the technical depth of the query."""
        advanced_terms = [
            "architecture",
            "pattern",
            "optimization",
            "algorithm",
            "complexity",
            "performance",
            "scalability",
            "design",
            "implementation",
            "framework",
        ]
        basic_terms = [
            "what",
            "how",
            "basic",
            "simple",
            "introduction",
            "beginner",
            "start",
            "getting started",
            "hello world",
        ]

        advanced_count = sum(1 for term in advanced_terms if term in query_lower)
        basic_count = sum(1 for term in basic_terms if term in query_lower)

        if advanced_count > basic_count and advanced_count > 0:
            return "advanced"
        if basic_count > 0:
            return "basic"
        return "medium"

    def _extract_entities(self, query_lower: str) -> list[str]:
        """Extract entities from the query (simple approach)."""
        # Simple entity extraction - could be enhanced with NER models
        entities = []

        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', query_lower)
        entities.extend(quoted_terms)

        # Extract terms in code blocks
        code_terms = re.findall(r"`([^`]*)`", query_lower)
        entities.extend(code_terms)

        # Extract function/method names
        function_names = re.findall(r"(\w+)\(\)", query_lower)
        entities.extend(function_names)

        return entities

    def _calculate_semantic_complexity(
        self, query_lower: str, _tokens: list[str]
    ) -> float:
        """Calculate semantic complexity of the query."""
        complexity_indicators = [
            "relationship",
            "difference",
            "comparison",
            "integration",
            "combination",
            "interaction",
            "dependency",
            "correlation",
            "cause",
            "effect",
            "consequence",
            "implication",
            "alternative",
            "trade-off",
        ]

        complexity_score = sum(
            1 for indicator in complexity_indicators if indicator in query_lower
        )
        return min(complexity_score / 5.0, 1.0)  # Normalize to 0-1

    def _calculate_keyword_density(self, _query_lower: str, tokens: list[str]) -> float:
        """Calculate technical keyword density."""
        if not tokens:
            return 0.0

        technical_tokens = sum(
            1
            for token in tokens
            if token in self._programming_keywords
            or token in self._programming_languages
        )

        return technical_tokens / len(tokens)
