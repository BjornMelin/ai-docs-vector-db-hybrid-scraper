"""Content type filtering for document classification and semantic filtering.

This module provides sophisticated content type filtering capabilities including
document type classification, semantic category filtering, intent-based filtering,
and content quality assessment.
"""

import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from qdrant_client import models

from .base import BaseFilter, FilterError, FilterResult


logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document types for classification."""

    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    PDF = "pdf"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    JUPYTER = "jupyter"
    DOCUMENTATION = "documentation"
    BLOG_POST = "blog_post"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    API_DOC = "api_doc"
    CHANGELOG = "changelog"
    README = "readme"
    LICENSE = "license"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ContentCategory(str, Enum):
    """Semantic categories for content classification."""

    PROGRAMMING = "programming"
    DEVELOPMENT = "development"
    DEVOPS = "devops"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"
    REFERENCE = "reference"
    BLOG = "blog"
    NEWS = "news"
    ACADEMIC = "academic"
    BUSINESS = "business"
    TECHNICAL = "technical"
    GENERAL = "general"
    RESEARCH = "research"
    GUIDE = "guide"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"
    EXAMPLES = "examples"
    COMPARISON = "comparison"
    REVIEW = "review"
    ANNOUNCEMENT = "announcement"


class ContentIntent(str, Enum):
    """Intent-based content classification."""

    LEARN = "learn"
    REFERENCE = "reference"
    TROUBLESHOOT = "troubleshoot"
    IMPLEMENT = "implement"
    UNDERSTAND = "understand"
    COMPARE = "compare"
    DECIDE = "decide"
    CONFIGURE = "configure"
    INSTALL = "install"
    DEBUG = "debug"
    OPTIMIZE = "optimize"
    MIGRATE = "migrate"
    INTEGRATE = "integrate"
    TEST = "test"
    DEPLOY = "deploy"


class QualityLevel(str, Enum):
    """Content quality levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ContentTypeCriteria(BaseModel):
    """Criteria for content type filtering operations."""

    # Document type filters
    document_types: list[DocumentType] | None = Field(
        None, description="Filter by specific document types"
    )
    exclude_document_types: list[DocumentType] | None = Field(
        None, description="Exclude specific document types"
    )

    # Content category filters
    categories: list[ContentCategory] | None = Field(
        None, description="Filter by content categories"
    )
    exclude_categories: list[ContentCategory] | None = Field(
        None, description="Exclude specific categories"
    )

    # Intent-based filters
    intents: list[ContentIntent] | None = Field(
        None, description="Filter by content intent"
    )
    exclude_intents: list[ContentIntent] | None = Field(
        None, description="Exclude specific content intents"
    )

    # Language and framework filters
    programming_languages: list[str] | None = Field(
        None, description="Filter by programming languages"
    )
    frameworks: list[str] | None = Field(
        None, description="Filter by frameworks or libraries"
    )

    # Quality filters
    min_quality_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum content quality score"
    )
    quality_levels: list[QualityLevel] | None = Field(
        None, description="Filter by quality levels"
    )

    # Content characteristics
    min_word_count: int | None = Field(None, ge=1, description="Minimum word count")
    max_word_count: int | None = Field(None, ge=1, description="Maximum word count")
    has_code_examples: bool | None = Field(
        None, description="Filter for content with code examples"
    )
    has_images: bool | None = Field(None, description="Filter for content with images")
    has_links: bool | None = Field(
        None, description="Filter for content with external links"
    )

    # Site and source filters
    site_names: list[str] | None = Field(
        None, description="Filter by specific site names"
    )
    exclude_sites: list[str] | None = Field(None, description="Exclude specific sites")
    crawl_sources: list[str] | None = Field(None, description="Filter by crawl sources")

    # Semantic filtering
    semantic_similarity_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum semantic similarity threshold"
    )
    semantic_keywords: list[str] | None = Field(
        None, description="Keywords for semantic matching"
    )

    @field_validator("min_word_count", "max_word_count")
    @classmethod
    def validate_word_counts(cls, v, info):
        """Validate word count ranges."""
        if v and v <= 0:
            raise ValueError("Word count must be positive")

        # Check min <= max if both are provided
        if (
            info.field_name == "max_word_count"
            and info.data.get("min_word_count")
            and v < info.data["min_word_count"]
        ):
            raise ValueError("max_word_count must be >= min_word_count")

        return v


class ContentClassification(BaseModel):
    """Result of content classification analysis."""

    document_type: DocumentType = Field(..., description="Classified document type")
    category: ContentCategory = Field(..., description="Content category")
    intent: ContentIntent = Field(..., description="Content intent")
    quality_level: QualityLevel = Field(..., description="Quality assessment")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    features: dict[str, Any] = Field(
        default_factory=dict, description="Extracted content features"
    )


class ContentTypeFilter(BaseFilter):
    """Advanced content type filtering with semantic classification."""

    def __init__(
        self,
        name: str = "content_type_filter",
        description: str = "Filter documents by type, category, intent, and semantic characteristics",
        enabled: bool = True,
        priority: int = 80,
    ):
        """Initialize content type filter.

        Args:
            name: Filter name
            description: Filter description
            enabled: Whether filter is enabled
            priority: Filter priority (higher = earlier execution)
        """
        super().__init__(name, description, enabled, priority)

        # Content field mappings for Qdrant
        self.content_fields = {
            "doc_type": "doc_type",
            "language": "language",
            "framework": "framework",
            "site_name": "site_name",
            "crawl_source": "crawl_source",
            "quality_score": "quality_score",
            "word_count": "word_count",
            "char_count": "char_count",
            "links_count": "links_count",
            "has_code": "has_code_examples",
            "has_images": "has_images",
            "content_category": "content_category",
            "content_intent": "content_intent",
        }

        # Programming language patterns
        self.language_patterns = {
            "python": r"\b(python|py|pip|conda|virtualenv|pytest|django|flask)\b",
            "javascript": r"\b(javascript|js|node|npm|yarn|react|vue|angular)\b",
            "typescript": r"\b(typescript|ts|tsc)\b",
            "java": r"\b(java|maven|gradle|spring|junit)\b",
            "csharp": r"\b(c#|csharp|dotnet|nuget|visual studio)\b",
            "cpp": r"\b(c\+\+|cpp|cmake|make|gcc|clang)\b",
            "rust": r"\b(rust|cargo|rustc)\b",
            "go": r"\b(golang|go|mod)\b",
            "php": r"\b(php|composer|laravel|symfony)\b",
            "ruby": r"\b(ruby|gem|rails|bundler)\b",
            "swift": r"\b(swift|xcode|cocoapods)\b",
            "kotlin": r"\b(kotlin|gradle)\b",
        }

    async def apply(
        self, filter_criteria: dict[str, Any], context: dict[str, Any] | None = None
    ) -> FilterResult:
        """Apply content type filtering with semantic analysis.

        Args:
            filter_criteria: Content type filter criteria
            context: Optional context with classification models and settings

        Returns:
            FilterResult with Qdrant content type filter conditions

        Raises:
            FilterError: If content type filter application fails
        """
        try:
            # Validate and parse criteria
            criteria = ContentTypeCriteria.model_validate(filter_criteria)

            # Build Qdrant filter conditions
            conditions = []
            metadata = {"applied_filters": [], "classification_info": {}}

            # Process document type filters
            doc_type_conditions = self._build_document_type_filters(criteria)
            conditions.extend(doc_type_conditions)
            if doc_type_conditions:
                metadata["applied_filters"].append("document_types")

            # Process category filters
            category_conditions = self._build_category_filters(criteria)
            conditions.extend(category_conditions)
            if category_conditions:
                metadata["applied_filters"].append("categories")

            # Process intent filters
            intent_conditions = self._build_intent_filters(criteria)
            conditions.extend(intent_conditions)
            if intent_conditions:
                metadata["applied_filters"].append("intents")

            # Process language and framework filters
            lang_conditions = self._build_language_filters(criteria)
            conditions.extend(lang_conditions)
            if lang_conditions:
                metadata["applied_filters"].append("languages")

            # Process quality filters
            quality_conditions = self._build_quality_filters(criteria)
            conditions.extend(quality_conditions)
            if quality_conditions:
                metadata["applied_filters"].append("quality")

            # Process content characteristic filters
            char_conditions = self._build_characteristic_filters(criteria)
            conditions.extend(char_conditions)
            if char_conditions:
                metadata["applied_filters"].append("characteristics")

            # Process site and source filters
            site_conditions = self._build_site_filters(criteria)
            conditions.extend(site_conditions)
            if site_conditions:
                metadata["applied_filters"].append("sites")

            # Process semantic filters
            semantic_conditions = self._build_semantic_filters(criteria, context)
            conditions.extend(semantic_conditions)
            if semantic_conditions:
                metadata["applied_filters"].append("semantic")

            # Calculate performance impact
            performance_impact = self._estimate_performance_impact(len(conditions))

            # Build final filter
            final_filter = None
            if conditions:
                final_filter = models.Filter(must=conditions)

                # Add classification metadata
                metadata["classification_info"] = {
                    "total_conditions": len(conditions),
                    "semantic_enabled": criteria.semantic_similarity_threshold
                    is not None,
                    "quality_filtering": criteria.min_quality_score is not None,
                }

            self._logger.info(
                f"Applied content type filter with {len(conditions)} conditions: "
                f"{metadata['applied_filters']}"
            )

            return FilterResult(
                filter_conditions=final_filter,
                metadata=metadata,
                confidence_score=0.90,
                performance_impact=performance_impact,
            )

        except Exception as e:
            error_msg = f"Failed to apply content type filter: {e}"
            self._logger.error(error_msg, exc_info=True)
            raise FilterError(
                error_msg,
                filter_name=self.name,
                filter_criteria=filter_criteria,
                underlying_error=e,
            ) from e

    def _build_document_type_filters(
        self, criteria: ContentTypeCriteria
    ) -> list[models.FieldCondition]:
        """Build filters for document types."""
        conditions = []

        # Include specific document types
        if criteria.document_types:
            doc_type_values = [dt.value for dt in criteria.document_types]
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["doc_type"],
                    match=models.MatchAny(any=doc_type_values),
                )
            )

        # Exclude specific document types
        if criteria.exclude_document_types:
            exclude_values = [dt.value for dt in criteria.exclude_document_types]
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["doc_type"],
                    match=models.MatchExcept(**{"except": exclude_values}),
                )
            )

        return conditions

    def _build_category_filters(
        self, criteria: ContentTypeCriteria
    ) -> list[models.FieldCondition]:
        """Build filters for content categories."""
        conditions = []

        # Include specific categories
        if criteria.categories:
            category_values = [cat.value for cat in criteria.categories]
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["content_category"],
                    match=models.MatchAny(any=category_values),
                )
            )

        # Exclude specific categories
        if criteria.exclude_categories:
            exclude_values = [cat.value for cat in criteria.exclude_categories]
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["content_category"],
                    match=models.MatchExcept(**{"except": exclude_values}),
                )
            )

        return conditions

    def _build_intent_filters(
        self, criteria: ContentTypeCriteria
    ) -> list[models.FieldCondition]:
        """Build filters for content intents."""
        conditions = []

        # Include specific intents
        if criteria.intents:
            intent_values = [intent.value for intent in criteria.intents]
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["content_intent"],
                    match=models.MatchAny(any=intent_values),
                )
            )

        # Exclude specific intents
        if criteria.exclude_intents:
            exclude_values = [intent.value for intent in criteria.exclude_intents]
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["content_intent"],
                    match=models.MatchExcept(**{"except": exclude_values}),
                )
            )

        return conditions

    def _build_language_filters(
        self, criteria: ContentTypeCriteria
    ) -> list[models.FieldCondition]:
        """Build filters for programming languages and frameworks."""
        conditions = []

        # Programming language filters
        if criteria.programming_languages:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["language"],
                    match=models.MatchAny(any=criteria.programming_languages),
                )
            )

        # Framework filters
        if criteria.frameworks:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["framework"],
                    match=models.MatchAny(any=criteria.frameworks),
                )
            )

        return conditions

    def _build_quality_filters(
        self, criteria: ContentTypeCriteria
    ) -> list[models.FieldCondition]:
        """Build filters for content quality."""
        conditions = []

        # Minimum quality score
        if criteria.min_quality_score is not None:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["quality_score"],
                    range=models.Range(gte=criteria.min_quality_score),
                )
            )

        # Quality levels (if stored as separate field)
        if criteria.quality_levels:
            quality_values = [ql.value for ql in criteria.quality_levels]
            conditions.append(
                models.FieldCondition(
                    key="quality_level", match=models.MatchAny(any=quality_values)
                )
            )

        return conditions

    def _build_characteristic_filters(
        self, criteria: ContentTypeCriteria
    ) -> list[models.FieldCondition]:
        """Build filters for content characteristics."""
        conditions = []

        # Word count filters
        if criteria.min_word_count is not None:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["word_count"],
                    range=models.Range(gte=criteria.min_word_count),
                )
            )

        if criteria.max_word_count is not None:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["word_count"],
                    range=models.Range(lte=criteria.max_word_count),
                )
            )

        # Content feature filters
        if criteria.has_code_examples is not None:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["has_code"],
                    match=models.MatchValue(value=criteria.has_code_examples),
                )
            )

        if criteria.has_images is not None:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["has_images"],
                    match=models.MatchValue(value=criteria.has_images),
                )
            )

        if criteria.has_links is not None:
            if criteria.has_links:
                # Has links: links_count > 0
                conditions.append(
                    models.FieldCondition(
                        key=self.content_fields["links_count"], range=models.Range(gt=0)
                    )
                )
            else:
                # No links: links_count = 0
                conditions.append(
                    models.FieldCondition(
                        key=self.content_fields["links_count"],
                        match=models.MatchValue(value=0),
                    )
                )

        return conditions

    def _build_site_filters(
        self, criteria: ContentTypeCriteria
    ) -> list[models.FieldCondition]:
        """Build filters for sites and sources."""
        conditions = []

        # Include specific sites
        if criteria.site_names:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["site_name"],
                    match=models.MatchAny(any=criteria.site_names),
                )
            )

        # Exclude specific sites
        if criteria.exclude_sites:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["site_name"],
                    match=models.MatchExcept(**{"except": criteria.exclude_sites}),
                )
            )

        # Crawl source filters
        if criteria.crawl_sources:
            conditions.append(
                models.FieldCondition(
                    key=self.content_fields["crawl_source"],
                    match=models.MatchAny(any=criteria.crawl_sources),
                )
            )

        return conditions

    def _build_semantic_filters(
        self, criteria: ContentTypeCriteria, _context: dict[str, Any] | None = None
    ) -> list[models.FieldCondition]:
        """Build semantic similarity filters."""
        conditions = []

        # Note: Semantic filtering typically requires vector similarity search
        # rather than payload filtering. This would be handled at a higher level
        # in the search orchestrator. Here we can add metadata filters that
        # support semantic search.

        if criteria.semantic_keywords:
            # Create text search conditions for semantic keywords
            keyword_conditions = []
            for keyword in criteria.semantic_keywords:
                keyword_conditions.append(
                    models.FieldCondition(
                        key="content_preview", match=models.MatchText(text=keyword)
                    )
                )

            # Use OR logic for keywords (should match at least one)
            if keyword_conditions:
                conditions.append(models.Filter(should=keyword_conditions))

        return conditions

    def _estimate_performance_impact(self, condition_count: int) -> str:
        """Estimate performance impact based on filter complexity."""
        if condition_count == 0:
            return "none"
        elif condition_count <= 3:
            return "low"
        elif condition_count <= 6:
            return "medium"
        else:
            return "high"

    async def validate_criteria(self, filter_criteria: dict[str, Any]) -> bool:
        """Validate content type filter criteria."""
        try:
            ContentTypeCriteria.model_validate(filter_criteria)
            return True
        except Exception as e:
            self._logger.warning(f"Invalid content type criteria: {e}")
            return False

    def get_supported_operators(self) -> list[str]:
        """Get supported content type operators."""
        return [
            "document_types",
            "exclude_document_types",
            "categories",
            "exclude_categories",
            "intents",
            "exclude_intents",
            "programming_languages",
            "frameworks",
            "min_quality_score",
            "quality_levels",
            "min_word_count",
            "max_word_count",
            "has_code_examples",
            "has_images",
            "has_links",
            "site_names",
            "exclude_sites",
            "crawl_sources",
            "semantic_keywords",
            "semantic_similarity_threshold",
        ]

    def classify_content(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> ContentClassification:
        """Classify content type, category, intent, and quality.

        Args:
            content: Text content to classify
            metadata: Optional metadata about the content

        Returns:
            ContentClassification with analysis results
        """
        # Basic classification based on content analysis
        # This is a simplified implementation - in production, you'd use
        # ML models or more sophisticated NLP techniques

        content_lower = content.lower()
        features = {}

        # Document type classification
        doc_type = self._classify_document_type(content, metadata)

        # Category classification
        category = self._classify_category(content_lower, metadata)

        # Intent classification
        intent = self._classify_intent(content_lower)

        # Quality assessment
        quality_level = self._assess_quality(content, metadata)

        # Extract features
        features.update(
            {
                "word_count": len(content.split()),
                "has_code": bool(
                    re.search(r"```|<code>|<pre>", content)
                    or re.search(
                        r"\bdef\s+\w+\s*\(|function\s+\w+\s*\(|class\s+\w+\s*[:\{]",
                        content,
                    )
                    or re.search(
                        r"import\s+\w+|from\s+\w+\s+import|#include\s*<", content
                    )
                    or re.search(
                        r"return\s+[^;]*;?\s*$|if\s*\([^)]*\)\s*[:\{]",
                        content,
                        re.MULTILINE,
                    )
                ),
                "has_links": bool(re.search(r"http[s]?://|www\.", content)),
                "programming_languages": self._detect_languages(content_lower),
            }
        )

        # Calculate confidence based on feature strength
        confidence = self._calculate_classification_confidence(features, metadata)

        return ContentClassification(
            document_type=doc_type,
            category=category,
            intent=intent,
            quality_level=quality_level,
            confidence=confidence,
            features=features,
        )

    def _classify_document_type(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> DocumentType:
        """Classify document type based on content and metadata."""
        if metadata and "doc_type" in metadata:
            try:
                return DocumentType(metadata["doc_type"])
            except ValueError:
                pass

        content_lower = content.lower()

        # Pattern-based classification
        # Look for code patterns first (most specific)
        if (
            re.search(r"```|<code>|<pre>", content)
            or re.search(
                r"\bdef\s+\w+\s*\(|function\s+\w+\s*\(|class\s+\w+\s*[:\{]", content
            )
            or re.search(r"import\s+\w+|from\s+\w+\s+import|#include\s*<", content)
            or re.search(
                r"return\s+[^;]*;?\s*$|if\s*\([^)]*\)\s*[:\{]", content, re.MULTILINE
            )
        ):
            return DocumentType.CODE
        elif re.search(r"#+\s+|^\*\s+|\[.*\]\(.*\)", content, re.MULTILINE):
            return DocumentType.MARKDOWN
        elif re.search(r"<html>|<body>|<div>", content):
            return DocumentType.HTML
        elif re.search(r"# getting started|# installation|# tutorial", content_lower):
            return DocumentType.TUTORIAL
        elif re.search(r"# api|# reference|# documentation", content_lower):
            return DocumentType.REFERENCE
        elif re.search(r"readme|read me", content_lower):
            return DocumentType.README

        return DocumentType.UNKNOWN

    def _classify_category(
        self, content_lower: str, _metadata: dict[str, Any] | None = None
    ) -> ContentCategory:
        """Classify content category."""
        # Check for programming/development indicators
        if any(lang in content_lower for lang in self.language_patterns):
            return ContentCategory.PROGRAMMING

        # Check for tutorial indicators
        if re.search(r"tutorial|how to|step by step|guide", content_lower):
            return ContentCategory.TUTORIAL

        # Check for documentation indicators
        if re.search(r"documentation|docs|reference|manual", content_lower):
            return ContentCategory.DOCUMENTATION

        # Check for troubleshooting
        if re.search(r"error|troubleshoot|fix|problem|issue", content_lower):
            return ContentCategory.TROUBLESHOOTING

        return ContentCategory.GENERAL

    def _classify_intent(self, content_lower: str) -> ContentIntent:
        """Classify content intent."""
        # Learn intent
        if re.search(r"learn|tutorial|introduction|getting started", content_lower):
            return ContentIntent.LEARN

        # Reference intent
        if re.search(r"reference|api|documentation|manual", content_lower):
            return ContentIntent.REFERENCE

        # Troubleshoot intent
        if re.search(r"troubleshoot|debug|fix|error|problem", content_lower):
            return ContentIntent.TROUBLESHOOT

        # Implementation intent
        if re.search(r"implement|build|create|develop|code", content_lower):
            return ContentIntent.IMPLEMENT

        # Configuration intent
        if re.search(r"configure|setup|install|deployment", content_lower):
            return ContentIntent.CONFIGURE

        return ContentIntent.UNDERSTAND

    def _assess_quality(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> QualityLevel:
        """Assess content quality based on various factors."""
        if metadata and "quality_score" in metadata:
            score = metadata["quality_score"]
            if score >= 0.7:
                return QualityLevel.HIGH
            elif score >= 0.4:
                return QualityLevel.MEDIUM
            else:
                return QualityLevel.LOW

        # Simple heuristic-based quality assessment
        word_count = len(content.split())
        has_structure = bool(re.search(r"#+\s+|<h[1-6]>", content))
        has_examples = bool(
            re.search(r"```|example|for instance", content, re.IGNORECASE)
        )

        quality_score = 0
        if word_count > 200:
            quality_score += 1
        if has_structure:
            quality_score += 1
        if has_examples:
            quality_score += 1

        if quality_score >= 2:
            return QualityLevel.HIGH
        elif quality_score == 1:
            return QualityLevel.MEDIUM
        else:
            return QualityLevel.LOW

    def _detect_languages(self, content_lower: str) -> list[str]:
        """Detect programming languages in content."""
        detected = []
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, content_lower):
                detected.append(lang)
        return detected

    def _calculate_classification_confidence(
        self, features: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> float:
        """Calculate confidence score for classification."""
        confidence = 0.5  # Base confidence

        # Boost confidence based on available features
        if features.get("word_count", 0) > 100:
            confidence += 0.1
        if features.get("has_code"):
            confidence += 0.1
        if features.get("programming_languages"):
            confidence += 0.1
        if metadata and "doc_type" in metadata:
            confidence += 0.2

        return min(1.0, confidence)
