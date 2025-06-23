"""Content quality assessment with multi-metric scoring system.

This module provides comprehensive quality assessment for extracted content,
including completeness, relevance, confidence, freshness, structure quality,
readability, and duplicate detection with similarity thresholds.
"""

import logging
import re
from datetime import datetime
from typing import Any

from .models import QualityScore


logger = logging.getLogger(__name__)


class QualityAssessor:
    """Multi-metric quality scoring system for content assessment."""

    def __init__(self, embedding_manager: Any = None):
        """Initialize quality assessor.

        Args:
            embedding_manager: Optional EmbeddingManager for semantic similarity
        """
        self.embedding_manager = embedding_manager
        self._initialized = False

        # Quality thresholds and weights
        self._quality_config = {
            "min_word_count": 10,
            "optimal_word_count": 300,
            "max_word_count": 5000,
            "min_sentence_count": 3,
            "optimal_sentence_count": 20,
            "max_paragraph_length": 500,
            "freshness_days_excellent": 7,
            "freshness_days_good": 30,
            "freshness_days_poor": 365,
            "similarity_threshold_duplicate": 0.85,
            "similarity_threshold_similar": 0.70,
        }

        # Content quality indicators
        self._quality_indicators = {
            "positive": [
                r"\b(comprehensive|detailed|thorough|complete)\b",
                r"\b(example|examples|illustration)\b",
                r"\b(step-by-step|walkthrough|tutorial)\b",
                r"\b(updated|latest|recent|new)\b",
                r"\b(official|documentation|guide)\b",
            ],
            "negative": [
                r"\b(under construction|coming soon|placeholder)\b",
                r"\b(todo|fixme|xxx)\b",
                r"\b(broken|error|failed|not working)\b",
                r"\b(outdated|deprecated|obsolete)\b",
                r"\btest\s+content\b",
            ],
            "structural": [
                r"^#{1,6}\s",  # Markdown headers
                r"^\d+\.",  # Numbered lists
                r"^[*\-+]\s",  # Bullet points
                r"```",  # Code blocks
                r"^>\s",  # Blockquotes
            ],
        }

        # Readability indicators
        self._readability_config = {
            "avg_sentence_length_optimal": 20,
            "avg_sentence_length_max": 40,
            "complex_words_threshold": 0.15,
            "passive_voice_threshold": 0.25,
        }

    async def initialize(self) -> None:
        """Initialize the quality assessor."""
        self._initialized = True
        logger.info("QualityAssessor initialized with multi-metric scoring")

    async def assess_quality(
        self,
        content: str,
        confidence_threshold: float = 0.8,
        query_context: str | None = None,
        extraction_metadata: dict[str, Any] | None = None,
        existing_content: list[str] | None = None,
    ) -> QualityScore:
        """Assess content quality using multiple metrics.

        Args:
            content: Content to assess
            confidence_threshold: Minimum confidence threshold (0-1)
            query_context: Optional query context for relevance scoring
            extraction_metadata: Optional metadata about extraction process
            existing_content: Optional list of existing content for duplicate detection

        Returns:
            QualityScore: Comprehensive quality assessment
        """
        if not self._initialized:
            await self.initialize()

        # Calculate individual quality metrics
        completeness = self._assess_completeness(content)
        relevance = await self._assess_relevance(content, query_context)
        confidence = self._assess_confidence(content, extraction_metadata)
        freshness = self._assess_freshness(content, extraction_metadata)
        structure_quality = self._assess_structure_quality(content)
        readability = self._assess_readability(content)
        duplicate_similarity = await self._assess_duplicate_similarity(
            content, existing_content
        )

        # Calculate overall score with weighted average
        weights = {
            "completeness": 0.25,
            "relevance": 0.20,
            "confidence": 0.20,
            "freshness": 0.10,
            "structure_quality": 0.10,
            "readability": 0.10,
            "duplicate_penalty": 0.05,  # Penalty for duplicates
        }

        overall_score = (
            completeness * weights["completeness"]
            + relevance * weights["relevance"]
            + confidence * weights["confidence"]
            + freshness * weights["freshness"]
            + structure_quality * weights["structure_quality"]
            + readability * weights["readability"]
            - duplicate_similarity * weights["duplicate_penalty"]
        )

        # Ensure score is in valid range
        overall_score = max(0.0, min(1.0, overall_score))

        # Check if meets quality threshold
        meets_threshold = overall_score >= confidence_threshold

        # Identify quality issues and suggestions
        quality_issues = self._identify_quality_issues(
            content, completeness, structure_quality, readability, duplicate_similarity
        )
        improvement_suggestions = self._generate_improvement_suggestions(
            content, completeness, structure_quality, readability, duplicate_similarity
        )

        return QualityScore(
            overall_score=overall_score,
            completeness=completeness,
            relevance=relevance,
            confidence=confidence,
            freshness=freshness,
            structure_quality=structure_quality,
            readability=readability,
            duplicate_similarity=duplicate_similarity,
            meets_threshold=meets_threshold,
            confidence_threshold=confidence_threshold,
            quality_issues=quality_issues,
            improvement_suggestions=improvement_suggestions,
        )

    def _assess_completeness(self, content: str) -> float:
        """Assess content completeness based on length and structure.

        Args:
            content: Content to assess

        Returns:
            float: Completeness score (0-1)
        """
        if not content.strip():
            return 0.0

        # Count basic metrics
        word_count = len(content.split())
        sentence_count = len(re.findall(r"[.!?]+", content))
        paragraph_count = len([p for p in content.split("\n\n") if p.strip()])

        # Score based on word count
        if word_count < self._quality_config["min_word_count"]:
            word_score = word_count / self._quality_config["min_word_count"] * 0.5
        elif word_count < self._quality_config["optimal_word_count"]:
            word_score = (
                0.5
                + (word_count - self._quality_config["min_word_count"])
                / (
                    self._quality_config["optimal_word_count"]
                    - self._quality_config["min_word_count"]
                )
                * 0.4
            )
        elif word_count <= self._quality_config["max_word_count"]:
            word_score = (
                0.9
                + (self._quality_config["max_word_count"] - word_count)
                / (
                    self._quality_config["max_word_count"]
                    - self._quality_config["optimal_word_count"]
                )
                * 0.1
            )
        else:
            # Penalize overly long content
            word_score = max(
                0.7,
                1.0
                - (word_count - self._quality_config["max_word_count"]) / 1000 * 0.1,
            )

        # Score based on sentence structure
        if sentence_count < self._quality_config["min_sentence_count"]:
            sentence_score = (
                sentence_count / self._quality_config["min_sentence_count"] * 0.5
            )
        else:
            avg_words_per_sentence = word_count / sentence_count
            # Optimal range: 10-25 words per sentence
            if 10 <= avg_words_per_sentence <= 25:
                sentence_score = 1.0
            elif avg_words_per_sentence < 10:
                sentence_score = 0.7 + (avg_words_per_sentence / 10) * 0.3
            else:
                sentence_score = max(0.5, 1.0 - (avg_words_per_sentence - 25) / 50)

        # Score based on paragraph structure
        if paragraph_count == 0:
            paragraph_score = 0.3
        elif paragraph_count == 1:
            paragraph_score = 0.6
        else:
            avg_words_per_paragraph = word_count / paragraph_count
            if avg_words_per_paragraph <= self._quality_config["max_paragraph_length"]:
                paragraph_score = 1.0
            else:
                paragraph_score = max(0.5, 1.0 - (avg_words_per_paragraph - 500) / 1000)

        # Weighted combination
        return word_score * 0.5 + sentence_score * 0.3 + paragraph_score * 0.2

    async def _assess_relevance(self, content: str, query_context: str | None) -> float:
        """Assess content relevance to query context.

        Args:
            content: Content to assess
            query_context: Optional query for relevance assessment

        Returns:
            float: Relevance score (0-1)
        """
        if not query_context:
            return 0.5  # Neutral score when no context provided

        try:
            # Use semantic similarity if embedding manager is available
            if self.embedding_manager:
                result = await self.embedding_manager.generate_embeddings(
                    texts=[content, query_context], quality_tier=None, auto_select=True
                )

                if (
                    result.get("success", False)
                    and len(result.get("embeddings", [])) == 2
                ):
                    embeddings = result["embeddings"]
                    similarity = self._cosine_similarity(embeddings[0], embeddings[1])
                    return max(0.0, min(1.0, similarity))

            # Fallback to keyword-based relevance
            content_lower = content.lower()
            query_lower = query_context.lower()

            # Extract keywords from query
            query_words = set(re.findall(r"\b\w+\b", query_lower))
            query_words = {
                word for word in query_words if len(word) > 2
            }  # Filter short words

            if not query_words:
                return 0.5

            # Count keyword matches
            matches = sum(1 for word in query_words if word in content_lower)
            relevance_score = min(matches / len(query_words), 1.0)

            # Boost score for exact phrase matches
            if query_lower in content_lower:
                relevance_score = min(relevance_score + 0.3, 1.0)

            return relevance_score

        except Exception as e:
            logger.warning(f"Relevance assessment failed: {e}")
            return 0.5

    def _assess_confidence(
        self, content: str, extraction_metadata: dict[str, Any] | None
    ) -> float:
        """Assess extraction confidence based on content and metadata.

        Args:
            content: Extracted content
            extraction_metadata: Metadata about extraction process

        Returns:
            float: Confidence score (0-1)
        """
        base_confidence = 0.7  # Start with reasonable baseline

        # Check for positive quality indicators
        positive_matches = sum(
            1
            for pattern in self._quality_indicators["positive"]
            if re.search(pattern, content, re.IGNORECASE)
        )
        base_confidence += min(positive_matches * 0.05, 0.2)

        # Penalize negative indicators
        negative_matches = sum(
            1
            for pattern in self._quality_indicators["negative"]
            if re.search(pattern, content, re.IGNORECASE)
        )
        base_confidence -= min(negative_matches * 0.1, 0.3)

        # Use extraction metadata if available
        if extraction_metadata:
            # Check extraction success indicators
            if extraction_metadata.get("success", True):
                base_confidence += 0.1

            # Check for extraction quality indicators
            quality_score = extraction_metadata.get("quality_score", 0.5)
            if isinstance(quality_score, int | float):
                base_confidence = (base_confidence + quality_score) / 2

            # Check for tier/method used
            tier_used = extraction_metadata.get("tier_used", "")
            if tier_used in ["crawl4ai", "playwright"]:
                base_confidence += 0.05
            elif tier_used == "lightweight":
                base_confidence -= 0.1

        # Check for content completeness indicators
        if len(content.split()) >= self._quality_config["optimal_word_count"]:
            base_confidence += 0.1

        return max(0.0, min(1.0, base_confidence))

    def _assess_freshness(
        self, content: str, extraction_metadata: dict[str, Any] | None
    ) -> float:
        """Assess content freshness based on timestamps and indicators.

        Args:
            content: Content to assess
            extraction_metadata: Metadata about extraction

        Returns:
            float: Freshness score (0-1)
        """
        now = datetime.now()

        # Try to extract date from metadata first
        if extraction_metadata:
            last_modified = extraction_metadata.get("last_modified")
            if last_modified:
                try:
                    if isinstance(last_modified, str):
                        # Try parsing common date formats
                        for fmt in [
                            "%Y-%m-%d",
                            "%Y-%m-%dT%H:%M:%S",
                            "%Y-%m-%d %H:%M:%S",
                        ]:
                            try:
                                mod_date = datetime.strptime(last_modified[:19], fmt)
                                days_old = (now - mod_date).days
                                return self._calculate_freshness_score(days_old)
                            except ValueError:
                                continue
                    elif isinstance(last_modified, datetime):
                        days_old = (now - last_modified).days
                        return self._calculate_freshness_score(days_old)
                except Exception as e:
                    logger.debug(f"Failed to parse last_modified date: {e}")

        # Try to extract dates from content
        date_patterns = [
            r"updated?\s*:?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            r"published?\s*:?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            r"last\s+modified?\s*:?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",  # Any date pattern
        ]

        latest_date = None
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    date_str = match.replace("/", "-")
                    parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if not latest_date or parsed_date > latest_date:
                        latest_date = parsed_date
                except ValueError:
                    continue

        if latest_date:
            days_old = (now - latest_date).days
            return self._calculate_freshness_score(days_old)

        # Check for freshness indicators in content
        fresh_indicators = [
            r"\b(today|yesterday|this week|recent|latest|new|updated)\b",
            r"\b(2024|2025)\b",  # Current/recent years
        ]

        fresh_matches = sum(
            1
            for pattern in fresh_indicators
            if re.search(pattern, content, re.IGNORECASE)
        )

        if fresh_matches > 0:
            return 0.7 + min(fresh_matches * 0.1, 0.3)

        # Default to neutral score
        return 0.5

    def _calculate_freshness_score(self, days_old: int) -> float:
        """Calculate freshness score based on age in days.

        Args:
            days_old: Age of content in days

        Returns:
            float: Freshness score (0-1)
        """
        if days_old < 0:
            return 1.0  # Future dates (edge case)
        elif days_old <= self._quality_config["freshness_days_excellent"]:
            return 1.0
        elif days_old <= self._quality_config["freshness_days_good"]:
            return (
                0.8
                - (days_old - self._quality_config["freshness_days_excellent"])
                / (
                    self._quality_config["freshness_days_good"]
                    - self._quality_config["freshness_days_excellent"]
                )
                * 0.3
            )
        elif days_old <= self._quality_config["freshness_days_poor"]:
            return (
                0.5
                - (days_old - self._quality_config["freshness_days_good"])
                / (
                    self._quality_config["freshness_days_poor"]
                    - self._quality_config["freshness_days_good"]
                )
                * 0.3
            )
        else:
            return max(
                0.1,
                0.2
                - (days_old - self._quality_config["freshness_days_poor"]) / 365 * 0.1,
            )

    def _assess_structure_quality(self, content: str) -> float:
        """Assess content structure and organization quality.

        Args:
            content: Content to assess

        Returns:
            float: Structure quality score (0-1)
        """
        if not content.strip():
            return 0.0

        score = 0.0

        # Check for structural elements
        structural_matches = sum(
            1
            for pattern in self._quality_indicators["structural"]
            if re.search(pattern, content, re.MULTILINE)
        )

        if structural_matches > 0:
            score += min(
                structural_matches / 5, 0.4
            )  # Up to 0.4 for structural elements

        # Check for proper paragraph breaks
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            score += 0.2

            # Check average paragraph length
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(
                paragraphs
            )
            if 30 <= avg_paragraph_length <= 150:  # Reasonable paragraph size
                score += 0.2

        # Check for balanced sentence lengths
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            length_variance = sum(
                (length - avg_length) ** 2 for length in sentence_lengths
            ) / len(sentence_lengths)

            # Prefer moderate sentence length with reasonable variation
            if 10 <= avg_length <= 25 and length_variance < 100:
                score += 0.2

        # Check for proper use of whitespace and formatting
        if "\n" in content:  # Has line breaks
            score += 0.1

        if re.search(r"\n\s*\n", content):  # Has paragraph breaks
            score += 0.1

        return min(score, 1.0)

    def _assess_readability(self, content: str) -> float:
        """Assess content readability and clarity.

        Args:
            content: Content to assess

        Returns:
            float: Readability score (0-1)
        """
        if not content.strip():
            return 0.0

        # Calculate basic readability metrics
        words = content.split()
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
            return 0.0

        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        if (
            avg_sentence_length
            <= self._readability_config["avg_sentence_length_optimal"]
        ):
            sentence_score = 1.0
        elif avg_sentence_length <= self._readability_config["avg_sentence_length_max"]:
            sentence_score = (
                1.0
                - (
                    avg_sentence_length
                    - self._readability_config["avg_sentence_length_optimal"]
                )
                / (
                    self._readability_config["avg_sentence_length_max"]
                    - self._readability_config["avg_sentence_length_optimal"]
                )
                * 0.5
            )
        else:
            sentence_score = max(
                0.2,
                0.5
                - (
                    avg_sentence_length
                    - self._readability_config["avg_sentence_length_max"]
                )
                / 20
                * 0.3,
            )

        # Complex word ratio (words with 3+ syllables, simplified)
        complex_words = sum(
            1 for word in words if len(re.findall(r"[aeiouAEIOU]", word)) >= 3
        )
        complex_ratio = complex_words / len(words) if words else 0

        if complex_ratio <= self._readability_config["complex_words_threshold"]:
            complexity_score = 1.0
        else:
            complexity_score = max(
                0.3,
                1.0
                - (complex_ratio - self._readability_config["complex_words_threshold"])
                / 0.2,
            )

        # Check for passive voice indicators (simplified)
        passive_indicators = re.findall(
            r"\b(was|were|been|being)\s+\w+ed\b", content, re.IGNORECASE
        )
        passive_ratio = len(passive_indicators) / len(sentences) if sentences else 0

        if passive_ratio <= self._readability_config["passive_voice_threshold"]:
            passive_score = 1.0
        else:
            passive_score = max(
                0.5,
                1.0
                - (passive_ratio - self._readability_config["passive_voice_threshold"])
                / 0.3,
            )

        # Combine scores
        return sentence_score * 0.4 + complexity_score * 0.4 + passive_score * 0.2

    async def _assess_duplicate_similarity(
        self, content: str, existing_content: list[str] | None
    ) -> float:
        """Assess similarity to existing content for duplicate detection.

        Args:
            content: Content to check
            existing_content: List of existing content to compare against

        Returns:
            float: Maximum similarity score (0-1, where 1 means identical)
        """
        if not existing_content:
            return 0.0

        try:
            # Use semantic similarity if embedding manager is available
            if self.embedding_manager and len(existing_content) > 0:
                all_texts = [content, *existing_content]

                result = await self.embedding_manager.generate_embeddings(
                    texts=all_texts, quality_tier=None, auto_select=True
                )

                if result.get("success", False) and len(
                    result.get("embeddings", [])
                ) == len(all_texts):
                    embeddings = result["embeddings"]
                    content_embedding = embeddings[0]
                    existing_embeddings = embeddings[1:]

                    max_similarity = 0.0
                    for existing_embedding in existing_embeddings:
                        similarity = self._cosine_similarity(
                            content_embedding, existing_embedding
                        )
                        max_similarity = max(max_similarity, similarity)

                    return max(0.0, max_similarity)

            # Fallback to simple text similarity
            content_words = set(content.lower().split())
            max_similarity = 0.0

            for existing in existing_content:
                existing_words = set(existing.lower().split())
                if content_words and existing_words:
                    intersection = content_words.intersection(existing_words)
                    union = content_words.union(existing_words)
                    jaccard_similarity = len(intersection) / len(union)
                    max_similarity = max(max_similarity, jaccard_similarity)

            return max_similarity

        except Exception as e:
            logger.warning(f"Duplicate similarity assessment failed: {e}")
            return 0.0

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity between vectors
        """
        try:
            import math

            dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0

    def _identify_quality_issues(
        self,
        content: str,
        completeness: float,
        structure_quality: float,
        readability: float,
        duplicate_similarity: float,
    ) -> list[str]:
        """Identify specific quality issues in content.

        Args:
            content: Content to analyze
            completeness: Completeness score
            structure_quality: Structure quality score
            readability: Readability score
            duplicate_similarity: Duplicate similarity score

        Returns:
            list[str]: List of identified quality issues
        """
        issues = []

        if completeness < 0.5:
            word_count = len(content.split())
            if word_count < self._quality_config["min_word_count"]:
                issues.append(
                    f"Content too short ({word_count} words, minimum {self._quality_config['min_word_count']})"
                )
            else:
                issues.append("Content appears incomplete")

        if structure_quality < 0.4:
            issues.append(
                "Poor content structure (missing headings, paragraphs, or formatting)"
            )

        if readability < 0.4:
            issues.append("Low readability (complex sentences or difficult vocabulary)")

        if (
            duplicate_similarity
            > self._quality_config["similarity_threshold_duplicate"]
        ):
            issues.append(
                f"High similarity to existing content ({duplicate_similarity:.2f})"
            )
        elif (
            duplicate_similarity > self._quality_config["similarity_threshold_similar"]
        ):
            issues.append(
                f"Moderate similarity to existing content ({duplicate_similarity:.2f})"
            )

        # Check for specific content issues
        negative_matches = sum(
            1
            for pattern in self._quality_indicators["negative"]
            if re.search(pattern, content, re.IGNORECASE)
        )
        if negative_matches > 0:
            issues.append("Content contains negative quality indicators")

        return issues

    def _generate_improvement_suggestions(
        self,
        content: str,
        completeness: float,
        structure_quality: float,
        readability: float,
        duplicate_similarity: float,
    ) -> list[str]:
        """Generate suggestions for content improvement.

        Args:
            content: Content to analyze
            completeness: Completeness score
            structure_quality: Structure quality score
            readability: Readability score
            duplicate_similarity: Duplicate similarity score

        Returns:
            list[str]: List of improvement suggestions
        """
        suggestions = []

        if completeness < 0.7:
            suggestions.append("Add more detailed content and examples")

        if structure_quality < 0.6:
            suggestions.append(
                "Improve content structure with headings and proper paragraphs"
            )

        if readability < 0.6:
            sentences = re.split(r"[.!?]+", content)
            avg_length = sum(len(s.split()) for s in sentences if s.strip()) / max(
                len([s for s in sentences if s.strip()]), 1
            )
            if avg_length > 25:
                suggestions.append("Use shorter sentences for better readability")
            suggestions.append("Simplify vocabulary and sentence structure")

        if duplicate_similarity > self._quality_config["similarity_threshold_similar"]:
            suggestions.append(
                "Add unique insights or perspectives to differentiate from similar content"
            )

        # Check for missing elements
        if not re.search(r"```", content) and (
            "code" in content.lower() or "function" in content.lower()
        ):
            suggestions.append("Consider adding code examples for better illustration")

        if not re.search(r"^#{1,6}\s", content, re.MULTILINE):
            suggestions.append("Add section headings to improve content organization")

        return suggestions

    async def cleanup(self) -> None:
        """Cleanup quality assessor resources."""
        self._initialized = False
        logger.info("QualityAssessor cleaned up")
