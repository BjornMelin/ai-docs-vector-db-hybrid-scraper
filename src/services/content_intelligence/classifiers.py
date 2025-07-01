"""Content type classification using local models and semantic analysis.

This module provides lightweight semantic analysis for content type detection
using local models to avoid external API dependencies. It combines rule-based
classification with semantic similarity for accurate content type detection.
"""

import logging
import math
import re
from typing import Any

from .models import ContentClassification, ContentType


logger = logging.getLogger(__name__)


# Classification constants
CLASSIFICATION_WEIGHTS = {
    "keyword_weight": 0.4,
    "url_weight": 0.3,
    "title_weight": 0.2,
    "indicator_weight": 0.1,
    "semantic_blend_ratio": 0.5,
    "code_boost_threshold": 0.8,
    "secondary_type_threshold": 0.2,
    "secondary_type_max_difference": 0.3,
    "min_classification_score": 0.1,
}


class ContentClassifier:
    """Lightweight content type classifier using local models and heuristics."""

    def __init__(self, embedding_manager: Any = None):
        """Initialize content classifier.

        Args:
            embedding_manager: Optional EmbeddingManager for semantic analysis

        """
        self.embedding_manager = embedding_manager
        self._initialized = False

        # Content type keywords and patterns for rule-based classification
        self._classification_patterns = {
            ContentType.DOCUMENTATION: {
                "keywords": [
                    "documentation",
                    "docs",
                    "guide",
                    "manual",
                    "specification",
                    "readme",
                    "getting started",
                    "overview",
                    "introduction",
                    "quickstart",
                    "user guide",
                    "developer guide",
                    "api docs",
                ],
                "url_patterns": [
                    r"/docs?/",
                    r"/documentation/",
                    r"/guide/",
                    r"/manual/",
                    r"/readme",
                    r"/wiki/",
                    r"/help/",
                ],
                "title_patterns": [
                    r"documentation",
                    r"docs",
                    r"guide",
                    r"manual",
                    r"readme",
                ],
                "content_indicators": [
                    "table of contents",
                    "installation",
                    "configuration",
                    "getting started",
                    "prerequisites",
                    "requirements",
                ],
            },
            ContentType.CODE: {
                "keywords": [
                    "function",
                    "class",
                    "method",
                    "variable",
                    "import",
                    "export",
                    "return",
                    "if",
                    "else",
                    "for",
                    "while",
                    "try",
                    "catch",
                    "def ",
                    "var ",
                    "let ",
                    "const ",
                    "public ",
                    "private ",
                ],
                "url_patterns": [
                    r"/src/",
                    r"/code/",
                    r"/examples?/",
                    r"/snippet/",
                    r"\.py$",
                    r"\.js$",
                    r"\.ts$",
                    r"\.java$",
                    r"\.cpp$",
                ],
                "content_indicators": [
                    "```",
                    "```python",
                    "```javascript",
                    "```java",
                    "```cpp",
                    "function(",
                    "class ",
                    "def ",
                    "import ",
                    "from ",
                    "// ",
                    "/* ",
                    "# ",
                    "<!-- ",
                ],
            },
            ContentType.FAQ: {
                "keywords": [
                    "faq",
                    "frequently asked",
                    "questions",
                    "q&a",
                    "question",
                    "answer",
                    "how to",
                    "why",
                    "what is",
                    "how do",
                ],
                "url_patterns": [
                    r"/faq",
                    r"/questions",
                    r"/q-?a",
                    r"/help/",
                    r"/support/",
                    r"/troubleshoot",
                ],
                "title_patterns": [r"faq", r"frequently asked", r"questions", r"q&a"],
                "content_indicators": [
                    "Q:",
                    "A:",
                    "Question:",
                    "Answer:",
                    "How to",
                    "Why ",
                    "What is",
                    "How do I",
                    "Can I",
                    "Is it possible",
                ],
            },
            ContentType.TUTORIAL: {
                "keywords": [
                    "tutorial",
                    "walkthrough",
                    "step by step",
                    "how to",
                    "lesson",
                    "course",
                    "training",
                    "learn",
                    "teaching",
                ],
                "url_patterns": [
                    r"/tutorial/",
                    r"/walkthrough/",
                    r"/lesson/",
                    r"/course/",
                    r"/learn/",
                    r"/training/",
                    r"/how-to/",
                ],
                "title_patterns": [
                    r"tutorial",
                    r"walkthrough",
                    r"how to",
                    r"step by step",
                ],
                "content_indicators": [
                    "step 1",
                    "step 2",
                    "first",
                    "next",
                    "then",
                    "finally",
                    "before you begin",
                    "prerequisites",
                    "what you'll learn",
                ],
            },
            ContentType.REFERENCE: {
                "keywords": [
                    "reference",
                    "api",
                    "specification",
                    "schema",
                    "parameters",
                    "methods",
                    "properties",
                    "endpoints",
                    "config",
                    "options",
                ],
                "url_patterns": [
                    r"/api/",
                    r"/reference/",
                    r"/spec/",
                    r"/schema/",
                    r"/config/",
                    r"/options/",
                ],
                "title_patterns": [r"api", r"reference", r"specification", r"config"],
                "content_indicators": [
                    "parameters",
                    "returns",
                    "type:",
                    "required",
                    "optional",
                    "example:",
                    "response:",
                    "request:",
                    "endpoint",
                ],
            },
            ContentType.BLOG: {
                "keywords": [
                    "blog",
                    "post",
                    "article",
                    "author",
                    "published",
                    "updated",
                    "tags",
                    "category",
                    "share",
                    "comment",
                ],
                "url_patterns": [
                    r"/blog/",
                    r"/post/",
                    r"/article/",
                    r"/news/",
                    r"/\d{4}/\d{2}/\d{2}/",
                    r"/author/",
                ],
                "content_indicators": [
                    "posted by",
                    "published on",
                    "written by",
                    "author:",
                    "read more",
                    "share this",
                    "comments",
                    "tags:",
                ],
            },
            ContentType.NEWS: {
                "keywords": [
                    "news",
                    "breaking",
                    "report",
                    "update",
                    "announcement",
                    "press release",
                    "latest",
                    "today",
                    "yesterday",
                ],
                "url_patterns": [
                    r"/news/",
                    r"/press/",
                    r"/updates?/",
                    r"/announcements?/",
                ],
                "content_indicators": [
                    "breaking:",
                    "update:",
                    "announced",
                    "reported",
                    "according to",
                    "sources say",
                    "press release",
                ],
            },
            ContentType.FORUM: {
                "keywords": [
                    "forum",
                    "discussion",
                    "thread",
                    "reply",
                    "post",
                    "topic",
                    "community",
                    "user",
                    "member",
                    "joined",
                ],
                "url_patterns": [
                    r"/forum/",
                    r"/discussion/",
                    r"/community/",
                    r"/thread/",
                ],
                "content_indicators": [
                    "replies:",
                    "views:",
                    "last post",
                    "thread",
                    "quote",
                    "originally posted",
                    "user joined",
                    "member since",
                ],
            },
        }

        # Programming language detection patterns
        self._programming_languages = {
            "python": [r"def \w+\(", r"import \w+", r"from \w+ import", r"class \w+:"],
            "javascript": [r"function \w+\(", r"const \w+ =", r"let \w+ =", r"=> \{"],
            "typescript": [r"interface \w+", r": string", r": number", r"export \w+"],
            "java": [r"public class", r"private \w+", r"public static void"],
            "cpp": [r"#include <", r"int main\(", r"std::", r"cout <<"],
            "csharp": [r"public class", r"namespace \w+", r"using System"],
            "go": [r"func \w+\(", r"package \w+", r"import \("],
            "rust": [r"fn \w+\(", r"let mut", r"use \w+"],
            "php": [r"<\?php", r"function \w+\(", r"\$\w+"],
            "ruby": [r"def \w+", r"class \w+", r"require "],
        }

    async def initialize(self) -> None:
        """Initialize the classifier (no additional setup needed)."""
        self._initialized = True
        logger.info("ContentClassifier initialized with rule-based patterns")

    async def classify_content(
        self,
        content: str,
        url: str | None = None,
        title: str | None = None,
        use_semantic_analysis: bool = True,
    ) -> ContentClassification:
        """Classify content type using local models and heuristics.

        Args:
            content: Text content to classify
            url: Optional URL for additional context
            title: Optional title for additional context
            use_semantic_analysis: Whether to use semantic similarity analysis

        Returns:
            ContentClassification: Classification results with confidence scores

        Raises:
            RuntimeError: If classifier not initialized

        """
        if not self._initialized:
            msg = "ContentClassifier not initialized"
            raise RuntimeError(msg)

        if not content.strip():
            return self._create_unknown_classification("Empty content provided")

        # Calculate rule-based scores
        type_scores, reasoning_parts = self._calculate_rule_based_scores(
            content, url, title
        )

        # Apply semantic analysis if requested
        if use_semantic_analysis and self.embedding_manager:
            type_scores = await self._apply_semantic_analysis(
                type_scores, content, reasoning_parts
            )

        # Apply contextual adjustments and special cases
        type_scores = self._apply_contextual_adjustments(
            type_scores, content, url, title
        )
        type_scores = self._handle_code_content_special_case(type_scores, content, url)

        # Determine final classification results
        return self._create_final_classification(type_scores, content, reasoning_parts)

    def _create_unknown_classification(self, reason: str) -> ContentClassification:
        """Create a classification result for unknown content."""
        return ContentClassification(
            primary_type=ContentType.UNKNOWN,
            secondary_types=[],
            confidence_scores={ContentType.UNKNOWN: 1.0},
            classification_reasoning=reason,
        )

    def _calculate_rule_based_scores(
        self, content: str, url: str | None, title: str | None
    ) -> tuple[dict[ContentType, float], list[str]]:
        """Calculate rule-based classification scores."""
        content_lower = content.lower()
        url_lower = url.lower() if url else ""
        title_lower = title.lower() if title else ""

        type_scores: dict[ContentType, float] = {}
        reasoning_parts = []

        for content_type, patterns in self._classification_patterns.items():
            score, type_reasoning = self._score_content_type(
                content_lower, url_lower, title_lower, patterns
            )
            type_scores[content_type] = score

            if score > CLASSIFICATION_WEIGHTS["min_classification_score"]:
                reasoning_parts.append(
                    f"{content_type.value}: {score:.2f} ({', '.join(type_reasoning)})"
                )

        return type_scores, reasoning_parts

    def _score_content_type(
        self, content_lower: str, url_lower: str, title_lower: str, patterns: dict
    ) -> tuple[float, list[str]]:
        """Score a single content type against patterns."""
        score = 0.0
        type_reasoning = []

        # Keyword matching
        keyword_matches = sum(
            1 for keyword in patterns["keywords"] if keyword in content_lower
        )
        if keyword_matches > 0:
            keyword_score = min(keyword_matches / len(patterns["keywords"]), 1.0)
            score += keyword_score * CLASSIFICATION_WEIGHTS["keyword_weight"]
            type_reasoning.append(
                f"keywords: {keyword_matches}/{len(patterns['keywords'])}"
            )

        # URL pattern matching
        if url_lower:
            url_matches = sum(
                1
                for pattern in patterns["url_patterns"]
                if re.search(pattern, url_lower)
            )
            if url_matches > 0:
                url_score = min(url_matches / len(patterns["url_patterns"]), 1.0)
                score += url_score * CLASSIFICATION_WEIGHTS["url_weight"]
                type_reasoning.append("URL patterns")

        # Title pattern matching
        if title_lower and patterns.get("title_patterns"):
            title_matches = sum(
                1
                for pattern in patterns["title_patterns"]
                if re.search(pattern, title_lower)
            )
            if title_matches > 0:
                title_score = min(title_matches / len(patterns["title_patterns"]), 1.0)
                score += title_score * CLASSIFICATION_WEIGHTS["title_weight"]
                type_reasoning.append("title patterns")

        # Content indicator matching
        if patterns.get("content_indicators"):
            indicator_matches = sum(
                1
                for indicator in patterns["content_indicators"]
                if indicator in content_lower
            )
            if indicator_matches > 0:
                indicator_score = min(
                    indicator_matches / len(patterns["content_indicators"]), 1.0
                )
                score += indicator_score * CLASSIFICATION_WEIGHTS["indicator_weight"]
                type_reasoning.append("indicators")

        return score, type_reasoning

    async def _apply_semantic_analysis(
        self,
        type_scores: dict[ContentType, float],
        content: str,
        reasoning_parts: list[str],
    ) -> dict[ContentType, float]:
        """Apply semantic analysis to enhance classification scores."""
        try:
            semantic_scores = await self._semantic_classification(content)
            blend_ratio = CLASSIFICATION_WEIGHTS["semantic_blend_ratio"]

            for content_type, score in type_scores.items():
                if content_type in semantic_scores:
                    type_scores[content_type] = (
                        score * (1.0 - blend_ratio)
                        + semantic_scores[content_type] * blend_ratio
                    )
            reasoning_parts.append("semantic analysis applied")
        except (OSError, PermissionError) as e:
            logger.warning("Semantic analysis failed")

        return type_scores

    def _handle_code_content_special_case(
        self, type_scores: dict[ContentType, float], content: str, url: str | None
    ) -> dict[ContentType, float]:
        """Handle special case for pure code content classification."""
        if not self._is_pure_code_content(content):
            return type_scores

        tutorial_url_boost = self._calculate_tutorial_url_boost(url)

        # Only boost CODE if tutorial score is low or no strong tutorial URL signal
        if type_scores.get(ContentType.TUTORIAL, 0) < 0.5 or tutorial_url_boost < 0.3:
            type_scores[ContentType.CODE] = max(
                type_scores.get(ContentType.CODE, 0),
                CLASSIFICATION_WEIGHTS["code_boost_threshold"],
            )

        return type_scores

    def _calculate_tutorial_url_boost(self, url: str | None) -> float:
        """Calculate tutorial boost based on URL patterns."""
        if not url:
            return 0.0

        url_lower = url.lower()
        tutorial_patterns = ["/tutorial/", "/how-to/", "/walkthrough/", "tutorial."]

        return (
            0.4 if any(pattern in url_lower for pattern in tutorial_patterns) else 0.0
        )

    def _create_final_classification(
        self,
        type_scores: dict[ContentType, float],
        content: str,
        reasoning_parts: list[str],
    ) -> ContentClassification:
        """Create the final classification result."""
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)

        if (
            not sorted_types
            or sorted_types[0][1] < CLASSIFICATION_WEIGHTS["min_classification_score"]
        ):
            primary_type = ContentType.UNKNOWN
            secondary_types = []
        else:
            primary_type = sorted_types[0][0]
            secondary_types = self._determine_secondary_types(sorted_types)

        # Generate reasoning
        reasoning = "Rule-based classification"
        if len(reasoning_parts) > 3:
            reasoning += f" (and {len(reasoning_parts) - 3} more factors)"

        return ContentClassification(
            primary_type=primary_type,
            secondary_types=secondary_types,
            confidence_scores=type_scores,
            classification_reasoning=reasoning,
            has_code_blocks=self._detect_code_blocks(content),
            programming_languages=self._detect_programming_languages(content),
            is_tutorial_like=self._is_tutorial_like(content),
            is_reference_like=self._is_reference_like(content),
        )

    def _determine_secondary_types(
        self, sorted_types: list[tuple[ContentType, float]]
    ) -> list[ContentType]:
        """Determine secondary content types from sorted scores."""
        return [
            content_type
            for content_type, score in sorted_types[1:]
            if (
                score > CLASSIFICATION_WEIGHTS["secondary_type_threshold"]
                and (sorted_types[0][1] - score)
                < CLASSIFICATION_WEIGHTS["secondary_type_max_difference"]
            )
        ]

    async def _semantic_classification(self, content: str) -> dict[ContentType, float]:
        """Perform semantic classification using embeddings.

        Args:
            content: Content to classify semantically

        Returns:
            dict[ContentType, float]: Semantic similarity scores for each type

        """
        # Reference texts for each content type (used for semantic similarity)
        reference_texts = {
            ContentType.DOCUMENTATION: "This is documentation that explains how to use a software system or API.",
            ContentType.CODE: "This is source code with functions, classes, and programming logic.",
            ContentType.FAQ: "This is a frequently asked questions section with questions and answers.",
            ContentType.TUTORIAL: "This is a step-by-step tutorial that teaches how to accomplish a task.",
            ContentType.REFERENCE: "This is an API reference with technical specifications and parameters.",
            ContentType.BLOG: "This is a blog post or article written by an author on a specific topic.",
            ContentType.NEWS: "This is a news article reporting on recent events or announcements.",
            ContentType.FORUM: "This is a forum discussion with posts, replies, and community interaction.",
        }

        try:
            # Get embeddings for content and reference texts
            all_texts = [content, *list(reference_texts.values())]

            # Use embedding manager to generate embeddings
            result = await self.embedding_manager.generate_embeddings(
                texts=all_texts,
                quality_tier=None,  # Use default
                auto_select=True,
            )

            if not result.get("success", False) or not result.get("embeddings"):
                logger.warning(
                    "Failed to generate embeddings for semantic classification"
                )
                return {}

            embeddings = result["embeddings"]
            content_embedding = embeddings[0]
            reference_embeddings = embeddings[1:]

            # Calculate cosine similarity
            scores = {}
            for i, (content_type, _) in enumerate(reference_texts.items()):
                similarity = self._cosine_similarity(
                    content_embedding, reference_embeddings[i]
                )
                scores[content_type] = max(0.0, similarity)  # Ensure non-negative

            return scores

        except (OSError, PermissionError) as e:
            logger.exception("Semantic classification failed")
            return {}

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity between vectors

        """
        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))

            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))

            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)
        except (OSError, PermissionError) as e:
            return 0.0

    def _detect_code_blocks(self, content: str) -> bool:
        """Detect if content contains code blocks.

        Args:
            content: Content to analyze

        Returns:
            bool: True if code blocks are detected

        """
        code_patterns = [
            r"```\w*\n",  # Markdown code fences
            r"^\s{4,}\w+",  # Indented code blocks
            r"<code>",  # HTML code tags
            r"<pre>",  # HTML pre tags
            r"function\s+\w+\s*\(",  # Function definitions
            r"class\s+\w+",  # Class definitions
            r"def\s+\w+\s*\(",  # Python function definitions
        ]

        return any(
            re.search(pattern, content, re.MULTILINE) for pattern in code_patterns
        )

    def _detect_programming_languages(self, content: str) -> list[str]:
        """Detect programming languages in content.

        Args:
            content: Content to analyze

        Returns:
            list[str]: List of detected programming languages

        """
        detected_languages = []

        for language, patterns in self._programming_languages.items():
            if any(re.search(pattern, content) for pattern in patterns):
                detected_languages.append(language)

        return detected_languages

    def _is_tutorial_like(self, content: str) -> bool:
        """Determine if content has tutorial characteristics.

        Args:
            content: Content to analyze

        Returns:
            bool: True if content appears tutorial-like

        """
        tutorial_indicators = [
            r"step\s+\d+",
            r"first",
            r"next",
            r"then",
            r"finally",
            r"before you begin",
            r"prerequisites",
            r"what you.ll learn",
            r"in this tutorial",
            r"follow these steps",
        ]

        matches = sum(
            1
            for pattern in tutorial_indicators
            if re.search(pattern, content, re.IGNORECASE)
        )

        return matches >= 2

    def _is_reference_like(self, content: str) -> bool:
        """Determine if content has reference characteristics.

        Args:
            content: Content to analyze

        Returns:
            bool: True if content appears reference-like

        """
        reference_indicators = [
            r"parameters?:",
            r"returns?:",
            r"type:",
            r"required",
            r"optional",
            r"example:",
            r"response:",
            r"request:",
            r"endpoint",
            r"method",
            r"property",
        ]

        matches = sum(
            1
            for pattern in reference_indicators
            if re.search(pattern, content, re.IGNORECASE)
        )

        return matches >= 3

    def _is_pure_code_content(self, content: str) -> bool:
        """Check if content is primarily code with minimal text.

        Args:
            content: Content to analyze

        Returns:
            bool: True if content appears to be pure code

        """
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if len(lines) < 3:
            return False

        # Count code-like vs text-like lines
        code_indicators = 0
        text_indicators = 0

        for line in lines:
            # Code indicators
            if any(
                pattern in line
                for pattern in [
                    "{",
                    "}",
                    "()",
                    "=>",
                    "function",
                    "class ",
                    "def ",
                    "const ",
                    "let ",
                    "var ",
                    "import ",
                    "from ",
                    "return ",
                    ";",
                    "//",
                    "/*",
                    "#",
                    "=",
                    "++",
                    "--",
                ]
            ):
                code_indicators += 1
            # Text indicators (natural language patterns)
            elif any(
                pattern in line.lower()
                for pattern in [
                    "the ",
                    "and ",
                    "that ",
                    "this ",
                    "with ",
                    "for ",
                    "you ",
                    "how ",
                    "what ",
                    "when ",
                    "where ",
                    "why ",
                    "can ",
                    "will ",
                    "should ",
                    "would ",
                ]
            ):
                text_indicators += 1

        # Pure code if majority of lines are code-like and minimal text
        return (
            code_indicators >= len(lines) * 0.6 and text_indicators < len(lines) * 0.2
        )

    def _apply_contextual_adjustments(
        self,
        type_scores: dict,
        content: str,
        url: str | None = None,
        title: str | None = None,
    ) -> dict:
        """Apply contextual adjustments to improve classification accuracy.

        Args:
            type_scores: Initial type scores
            content: Content text
            url: Optional URL
            title: Optional title

        Returns:
            dict: Adjusted type scores

        """
        content_lower = content.lower()

        # Forum vs Code disambiguation: forums often contain code but have discussion patterns
        if (
            type_scores.get(ContentType.CODE, 0) > 0.3
            and type_scores.get(ContentType.FORUM, 0) > 0.1
        ):
            # Look for strong forum indicators
            forum_indicators = [
                "reply",
                "post",
                "thread",
                "forum",
                "discussion",
                "user joined",
                "member since",
                "views:",
                "replies:",
                "quote",
                "originally posted",
            ]
            forum_score = sum(
                1 for indicator in forum_indicators if indicator in content_lower
            )

            # If strong forum context, boost forum and reduce code
            if forum_score >= 3:
                type_scores[ContentType.FORUM] += 0.2
                type_scores[ContentType.CODE] *= 0.7

        # Tutorial vs Code disambiguation: tutorials teach but contain code examples
        if (
            type_scores.get(ContentType.CODE, 0) > 0.3
            and type_scores.get(ContentType.TUTORIAL, 0) > 0.1
        ):
            tutorial_indicators = [
                "step",
                "first",
                "next",
                "then",
                "finally",
                "learn",
                "tutorial",
                "walkthrough",
                "how to",
                "let's",
                "we'll",
                "you'll",
            ]
            tutorial_score = sum(
                1 for indicator in tutorial_indicators if indicator in content_lower
            )

            if tutorial_score >= 2:
                type_scores[ContentType.TUTORIAL] += 0.15
                type_scores[ContentType.CODE] *= 0.8

        # URL-based strong signals
        if url:
            url_lower = url.lower()
            if any(
                pattern in url_lower
                for pattern in ["/forum/", "/discussion/", "forum.", "/thread/"]
            ):
                type_scores[ContentType.FORUM] += 0.4
            elif any(
                pattern in url_lower
                for pattern in ["/tutorial/", "/how-to/", "/walkthrough/", "tutorial."]
            ):
                type_scores[ContentType.TUTORIAL] += 0.4
            elif "/news/" in url_lower:
                type_scores[ContentType.NEWS] += 0.3

        # Title-based adjustments
        if title:
            title_lower = title.lower()
            if any(word in title_lower for word in ["forum", "discussion", "thread"]):
                type_scores[ContentType.FORUM] += 0.2
            elif any(
                word in title_lower for word in ["tutorial", "how to", "walkthrough"]
            ):
                type_scores[ContentType.TUTORIAL] += 0.2

        # Ensure scores don't exceed 1.0
        for content_type, score in type_scores.items():
            type_scores[content_type] = min(score, 1.0)

        return type_scores

    async def cleanup(self) -> None:
        """Cleanup classifier resources."""
        self._initialized = False
        logger.info("ContentClassifier cleaned up")
