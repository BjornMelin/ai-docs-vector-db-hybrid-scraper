"""Personalized ranking service for adaptive search result ranking.

This module provides sophisticated personalized ranking capabilities that learn from
user interactions, preferences, and behavior patterns to deliver customized search
result rankings optimized for individual users and contexts.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


class RankingStrategy(str, Enum):
    """Ranking strategies for personalized results."""

    COLLABORATIVE_FILTERING = "collaborative_filtering"  # User-based similarities
    CONTENT_BASED = "content_based"  # Content feature matching
    HYBRID = "hybrid"  # Combined approach
    LEARNING_TO_RANK = "learning_to_rank"  # ML-based ranking
    BEHAVIORAL = "behavioral"  # Behavior pattern based
    CONTEXTUAL = "contextual"  # Context-aware ranking
    DEFAULT = "default"  # Non-personalized baseline


class InteractionType(str, Enum):
    """Types of user interactions with search results."""

    CLICK = "click"  # User clicked on result
    VIEW = "view"  # User viewed result details
    DOWNLOAD = "download"  # User downloaded content
    BOOKMARK = "bookmark"  # User saved/bookmarked
    SHARE = "share"  # User shared the result
    DWELL_TIME = "dwell_time"  # Time spent on result
    RATING = "rating"  # Explicit user rating
    SKIP = "skip"  # User skipped/ignored result
    NEGATIVE_FEEDBACK = "negative_feedback"  # User marked as irrelevant


class ContentCategory(str, Enum):
    """Content categories for preference modeling."""

    PROGRAMMING = "programming"
    DOCUMENTATION = "documentation"
    TUTORIAL = "tutorial"
    BLOG_POST = "blog_post"
    ACADEMIC = "academic"
    NEWS = "news"
    REFERENCE = "reference"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"
    EXAMPLES = "examples"
    TOOLS = "tools"
    FRAMEWORKS = "frameworks"


class UserPreference(BaseModel):
    """User preference for specific content attributes."""

    attribute: str = Field(..., description="Preference attribute name")
    value: str | float = Field(..., description="Preferred value")
    weight: float = Field(..., ge=0.0, le=1.0, description="Preference strength")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in preference"
    )
    learned_from: list[str] = Field(
        default_factory=list, description="Interaction types that contributed"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


class InteractionEvent(BaseModel):
    """User interaction event with search results."""

    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    result_id: str = Field(..., description="Search result identifier")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Interaction timestamp"
    )

    # Interaction details
    value: float | None = Field(
        None, description="Numeric value (rating, dwell time, etc.)"
    )
    query: str | None = Field(None, description="Search query context")
    result_position: int | None = Field(None, ge=0, description="Position in results")

    # Context information
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )

    @field_validator("value")
    @classmethod
    def validate_interaction_value(cls, v, info):
        """Validate interaction value based on type."""
        interaction_type = info.data.get("interaction_type")

        if interaction_type == InteractionType.RATING:
            if v is not None and not (1.0 <= v <= 5.0):
                msg = "Rating must be between 1.0 and 5.0"
                raise ValueError(msg)
        elif interaction_type == InteractionType.DWELL_TIME and v is not None and v < 0:
            msg = "Dwell time cannot be negative"
            raise ValueError(msg)

        return v


class UserProfile(BaseModel):
    """Comprehensive user profile for personalization."""

    user_id: str = Field(..., description="Unique user identifier")
    preferences: list[UserPreference] = Field(
        default_factory=list, description="Learned user preferences"
    )

    # Behavioral patterns
    avg_session_length: float = Field(
        0.0, ge=0.0, description="Average session duration"
    )
    preferred_result_types: dict[str, float] = Field(
        default_factory=dict, description="Preferred content types"
    )
    query_patterns: dict[str, int] = Field(
        default_factory=dict, description="Query frequency patterns"
    )

    # Temporal patterns
    active_hours: dict[int, float] = Field(
        default_factory=dict, description="Activity by hour of day"
    )
    interaction_velocity: float = Field(
        0.0, ge=0.0, description="Interactions per session"
    )

    # Quality indicators
    exploration_tendency: float = Field(
        0.5, ge=0.0, le=1.0, description="Tendency to explore new content"
    )
    quality_sensitivity: float = Field(
        0.5, ge=0.0, le=1.0, description="Sensitivity to content quality"
    )

    # Profile metadata
    total_interactions: int = Field(0, ge=0, description="Total interaction count")
    profile_created: datetime = Field(
        default_factory=datetime.now, description="Profile creation date"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last profile update"
    )
    confidence_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Profile confidence"
    )


class RankedResult(BaseModel):
    """Search result with personalized ranking information."""

    result_id: str = Field(..., description="Unique result identifier")
    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result content or snippet")

    # Ranking scores
    original_score: float = Field(
        ..., ge=0.0, le=1.0, description="Original relevance score"
    )
    personalized_score: float = Field(
        ..., ge=0.0, le=1.0, description="Personalized score"
    )
    final_score: float = Field(..., ge=0.0, le=1.0, description="Final ranking score")

    # Ranking factors
    ranking_factors: dict[str, float] = Field(
        default_factory=dict, description="Individual ranking factor contributions"
    )
    personalization_boost: float = Field(
        0.0, ge=-1.0, le=1.0, description="Personalization adjustment"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional result metadata"
    )
    ranking_explanation: str | None = Field(
        None, description="Human-readable ranking explanation"
    )


class PersonalizedRankingRequest(BaseModel):
    """Request for personalized result ranking."""

    # Core ranking data
    user_id: str = Field(..., description="User identifier")
    session_id: str | None = Field(None, description="Session identifier")
    query: str = Field(..., description="Search query")
    results: list[dict[str, Any]] = Field(..., description="Results to rank")

    # Ranking configuration
    strategy: RankingStrategy = Field(
        RankingStrategy.HYBRID, description="Ranking strategy"
    )
    personalization_strength: float = Field(
        0.7, ge=0.0, le=1.0, description="Strength of personalization"
    )

    # Context information
    context: dict[str, Any] = Field(
        default_factory=dict, description="Query and session context"
    )

    # Quality controls
    min_confidence_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="Minimum confidence for personalization"
    )
    diversity_factor: float = Field(
        0.1, ge=0.0, le=1.0, description="Result diversity promotion factor"
    )
    freshness_factor: float = Field(
        0.1, ge=0.0, le=1.0, description="Content freshness importance"
    )

    # Processing options
    enable_explanations: bool = Field(
        False, description="Generate ranking explanations"
    )
    enable_ab_testing: bool = Field(False, description="Enable A/B testing variations")
    max_processing_time_ms: float = Field(
        1000.0, ge=100.0, description="Maximum processing time"
    )

    @field_validator("results")
    @classmethod
    def validate_results_structure(cls, v):
        """Validate results have required fields."""
        if not v:
            msg = "Results list cannot be empty"
            raise ValueError(msg)

        required_fields = {"id", "title", "score"}
        for result in v:
            if not all(field in result for field in required_fields):
                msg = f"Results must contain fields: {required_fields}"
                raise ValueError(msg)

        return v


class PersonalizedRankingResult(BaseModel):
    """Result of personalized ranking operations."""

    ranked_results: list[RankedResult] = Field(
        ..., description="Personalized ranked results"
    )

    # Ranking metadata
    strategy_used: RankingStrategy = Field(..., description="Strategy applied")
    personalization_applied: bool = Field(
        ..., description="Whether personalization was used"
    )
    user_profile_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="User profile confidence"
    )

    # Performance metrics
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time")
    reranking_impact: float = Field(..., ge=0.0, description="Average position change")

    # Quality metrics
    diversity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Result diversity score"
    )
    coverage_score: float = Field(
        ..., ge=0.0, le=1.0, description="User preference coverage"
    )

    # Analysis metadata
    ranking_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Detailed ranking information"
    )


class PersonalizedRankingService:
    """Advanced personalized ranking service with machine learning."""

    def __init__(
        self,
        enable_learning: bool = True,
        enable_collaborative_filtering: bool = True,
        profile_cache_size: int = 1000,
        interaction_retention_days: int = 90,
    ):
        """Initialize personalized ranking service.

        Args:
            enable_learning: Enable learning from user interactions
            enable_collaborative_filtering: Enable collaborative filtering features
            profile_cache_size: Size of user profile cache
            interaction_retention_days: Days to retain interaction data

        """
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Configuration
        self.enable_learning = enable_learning
        self.enable_collaborative_filtering = enable_collaborative_filtering
        self.interaction_retention_days = interaction_retention_days

        # User profiles and interactions
        self.user_profiles = {}  # Cache of user profiles
        self.interaction_history = {}  # Recent interaction history
        self.profile_cache_size = profile_cache_size

        # Collaborative filtering data
        self.user_similarity_matrix = {}
        self.item_features = {}

        # Learning models (simplified implementations)
        self.content_embeddings = {}
        self.ranking_models = {}

        # Performance tracking
        self.performance_stats = {
            "total_rankings": 0,
            "avg_processing_time": 0.0,
            "personalization_rate": 0.0,
            "strategy_usage": {},
        }

    async def rank_results(
        self, request: PersonalizedRankingRequest
    ) -> PersonalizedRankingResult:
        """Apply personalized ranking to search results.

        Args:
            request: Ranking request with user context and results

        Returns:
            PersonalizedRankingResult with reranked results and metadata

        """
        start_time = time.time()

        try:
            # Get or create user profile
            user_profile = await self._get_user_profile(request.user_id)

            # Check if personalization should be applied
            should_personalize = self._should_apply_personalization(
                user_profile, request
            )

            if should_personalize:
                # Apply personalized ranking
                ranked_results = await self._apply_personalized_ranking(
                    request, user_profile
                )
                strategy_used = request.strategy
                personalization_applied = True
            else:
                # Fall back to default ranking
                ranked_results = self._apply_default_ranking(request.results)
                strategy_used = RankingStrategy.DEFAULT
                personalization_applied = False

            # Calculate quality metrics
            diversity_score = self._calculate_diversity_score(ranked_results)
            coverage_score = self._calculate_coverage_score(
                ranked_results, user_profile
            )
            reranking_impact = self._calculate_reranking_impact(
                request.results, ranked_results
            )

            processing_time_ms = (time.time() - start_time) * 1000

            # Build result
            result = PersonalizedRankingResult(
                ranked_results=ranked_results,
                strategy_used=strategy_used,
                personalization_applied=personalization_applied,
                user_profile_confidence=user_profile.confidence_score,
                processing_time_ms=processing_time_ms,
                reranking_impact=reranking_impact,
                diversity_score=diversity_score,
                coverage_score=coverage_score,
                ranking_metadata={
                    "user_preferences_count": len(user_profile.preferences),
                    "total_interactions": user_profile.total_interactions,
                    "personalization_strength": request.personalization_strength,
                    "factors_applied": self._get_applied_factors(request.strategy),
                },
            )

            # Update performance stats
            self._update_performance_stats(
                strategy_used, processing_time_ms, personalization_applied
            )

            self._logger.info(
                f"Ranked {len(ranked_results)} results for user {request.user_id} "
                f"using {strategy_used.value} in {processing_time_ms:.1f}ms "
                f"(personalized: {personalization_applied})"
            )

            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._logger.error(f"Personalized ranking failed: {e}", exc_info=True)  # TODO: Convert f-string to logging format

            # Return fallback result with default ranking
            fallback_results = self._apply_default_ranking(request.results)
            return PersonalizedRankingResult(
                ranked_results=fallback_results,
                strategy_used=RankingStrategy.DEFAULT,
                personalization_applied=False,
                user_profile_confidence=0.0,
                processing_time_ms=processing_time_ms,
                reranking_impact=0.0,
                diversity_score=0.5,
                coverage_score=0.0,
                ranking_metadata={"error": str(e)},
            )

    async def record_interaction(self, interaction: InteractionEvent) -> None:
        """Record user interaction for learning.

        Args:
            interaction: User interaction event

        """
        try:
            if not self.enable_learning:
                return

            # Store interaction
            user_id = interaction.user_id
            if user_id not in self.interaction_history:
                self.interaction_history[user_id] = []

            self.interaction_history[user_id].append(interaction)

            # Cleanup old interactions
            cutoff_date = datetime.now(tz=UTC) - timedelta(
                days=self.interaction_retention_days
            )
            self.interaction_history[user_id] = [
                i
                for i in self.interaction_history[user_id]
                if i.timestamp > cutoff_date
            ]

            # Update user profile asynchronously
            await self._update_user_profile_from_interaction(interaction)

            self._logger.debug(
                f"Recorded {interaction.interaction_type.value} interaction "
                f"for user {user_id} on result {interaction.result_id}"
            )

        except Exception as e:
            self._logger.error(f"Failed to record interaction: {e}", exc_info=True)  # TODO: Convert f-string to logging format

    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        # Create new profile
        profile = UserProfile(user_id=user_id)

        # Load existing interaction data if available
        if user_id in self.interaction_history:
            await self._build_profile_from_interactions(profile, user_id)

        # Cache profile
        if len(self.user_profiles) >= self.profile_cache_size:
            # Simple LRU eviction
            oldest_user = min(
                self.user_profiles.keys(),
                key=lambda u: self.user_profiles[u].last_updated,
            )
            del self.user_profiles[oldest_user]

        self.user_profiles[user_id] = profile
        return profile

    def _should_apply_personalization(
        self, user_profile: UserProfile, request: PersonalizedRankingRequest
    ) -> bool:
        """Determine if personalization should be applied."""
        # Check minimum confidence threshold
        if user_profile.confidence_score < request.min_confidence_threshold:
            return False

        # Check if user has sufficient interaction history
        if user_profile.total_interactions < 5:
            return False

        # Check strategy compatibility
        return request.strategy != RankingStrategy.DEFAULT

    async def _apply_personalized_ranking(
        self, request: PersonalizedRankingRequest, user_profile: UserProfile
    ) -> list[RankedResult]:
        """Apply personalized ranking based on strategy."""
        results = request.results

        if request.strategy == RankingStrategy.CONTENT_BASED:
            return await self._content_based_ranking(results, user_profile, request)
        if request.strategy == RankingStrategy.COLLABORATIVE_FILTERING:
            return await self._collaborative_filtering_ranking(
                results, user_profile, request
            )
        if request.strategy == RankingStrategy.BEHAVIORAL:
            return await self._behavioral_ranking(results, user_profile, request)
        if request.strategy == RankingStrategy.CONTEXTUAL:
            return await self._contextual_ranking(results, user_profile, request)
        if request.strategy == RankingStrategy.LEARNING_TO_RANK:
            return await self._learning_to_rank(results, user_profile, request)
        # HYBRID
        return await self._hybrid_ranking(results, user_profile, request)

    async def _content_based_ranking(
        self,
        results: list[dict[str, Any]],
        user_profile: UserProfile,
        request: PersonalizedRankingRequest,
    ) -> list[RankedResult]:
        """Apply content-based personalized ranking."""
        ranked_results = []

        for result in results:
            original_score = float(result.get("score", 0.0))

            # Calculate content-based personalization boost
            content_boost = self._calculate_content_preference_boost(
                result, user_profile
            )

            # Apply personalization strength
            personalization_boost = content_boost * request.personalization_strength

            # Calculate final score
            final_score = min(1.0, original_score + personalization_boost)

            # Build ranking factors
            ranking_factors = {
                "original_relevance": original_score,
                "content_preference": content_boost,
                "personalization_strength": request.personalization_strength,
            }

            ranked_result = RankedResult(
                result_id=result["id"],
                title=result["title"],
                content=result.get("content", ""),
                original_score=original_score,
                personalized_score=original_score + content_boost,
                final_score=final_score,
                ranking_factors=ranking_factors,
                personalization_boost=personalization_boost,
                metadata=result,
                ranking_explanation=(
                    self._generate_content_explanation(content_boost, user_profile)
                    if request.enable_explanations
                    else None
                ),
            )

            ranked_results.append(ranked_result)

        # Sort by final score
        ranked_results.sort(key=lambda r: r.final_score, reverse=True)

        return ranked_results

    async def _collaborative_filtering_ranking(
        self,
        results: list[dict[str, Any]],
        user_profile: UserProfile,
        request: PersonalizedRankingRequest,
    ) -> list[RankedResult]:
        """Apply collaborative filtering ranking."""
        if not self.enable_collaborative_filtering:
            return self._apply_default_ranking(results)

        ranked_results = []

        # Find similar users
        similar_users = self._find_similar_users(user_profile.user_id)

        for result in results:
            original_score = float(result.get("score", 0.0))

            # Calculate collaborative filtering boost
            collab_boost = self._calculate_collaborative_boost(result, similar_users)

            # Apply personalization strength
            personalization_boost = collab_boost * request.personalization_strength
            final_score = min(1.0, original_score + personalization_boost)

            ranking_factors = {
                "original_relevance": original_score,
                "collaborative_signal": collab_boost,
                "similar_users_count": len(similar_users),
            }

            ranked_result = RankedResult(
                result_id=result["id"],
                title=result["title"],
                content=result.get("content", ""),
                original_score=original_score,
                personalized_score=original_score + collab_boost,
                final_score=final_score,
                ranking_factors=ranking_factors,
                personalization_boost=personalization_boost,
                metadata=result,
            )

            ranked_results.append(ranked_result)

        ranked_results.sort(key=lambda r: r.final_score, reverse=True)
        return ranked_results

    async def _behavioral_ranking(
        self,
        results: list[dict[str, Any]],
        user_profile: UserProfile,
        request: PersonalizedRankingRequest,
    ) -> list[RankedResult]:
        """Apply behavioral pattern-based ranking."""
        ranked_results = []

        for result in results:
            original_score = float(result.get("score", 0.0))

            # Calculate behavioral boost based on user patterns
            behavioral_boost = self._calculate_behavioral_boost(
                result, user_profile, request.context
            )

            personalization_boost = behavioral_boost * request.personalization_strength
            final_score = min(1.0, original_score + personalization_boost)

            ranking_factors = {
                "original_relevance": original_score,
                "behavioral_pattern": behavioral_boost,
                "time_preference": self._get_time_preference_boost(user_profile),
                "query_pattern_match": self._get_query_pattern_boost(
                    request.query, user_profile
                ),
            }

            ranked_result = RankedResult(
                result_id=result["id"],
                title=result["title"],
                content=result.get("content", ""),
                original_score=original_score,
                personalized_score=original_score + behavioral_boost,
                final_score=final_score,
                ranking_factors=ranking_factors,
                personalization_boost=personalization_boost,
                metadata=result,
            )

            ranked_results.append(ranked_result)

        ranked_results.sort(key=lambda r: r.final_score, reverse=True)
        return ranked_results

    async def _contextual_ranking(
        self,
        results: list[dict[str, Any]],
        user_profile: UserProfile,
        request: PersonalizedRankingRequest,
    ) -> list[RankedResult]:
        """Apply context-aware ranking."""
        ranked_results = []
        context = request.context

        for result in results:
            original_score = float(result.get("score", 0.0))

            # Calculate contextual boost
            contextual_boost = self._calculate_contextual_boost(
                result, user_profile, context
            )

            personalization_boost = contextual_boost * request.personalization_strength
            final_score = min(1.0, original_score + personalization_boost)

            ranking_factors = {
                "original_relevance": original_score,
                "contextual_relevance": contextual_boost,
                "session_context": self._get_session_context_boost(context),
                "temporal_context": self._get_temporal_context_boost(context),
            }

            ranked_result = RankedResult(
                result_id=result["id"],
                title=result["title"],
                content=result.get("content", ""),
                original_score=original_score,
                personalized_score=original_score + contextual_boost,
                final_score=final_score,
                ranking_factors=ranking_factors,
                personalization_boost=personalization_boost,
                metadata=result,
            )

            ranked_results.append(ranked_result)

        ranked_results.sort(key=lambda r: r.final_score, reverse=True)
        return ranked_results

    async def _learning_to_rank(
        self,
        results: list[dict[str, Any]],
        user_profile: UserProfile,
        request: PersonalizedRankingRequest,
    ) -> list[RankedResult]:
        """Apply machine learning-based ranking."""
        # Simplified ML ranking implementation
        ranked_results = []

        for result in results:
            original_score = float(result.get("score", 0.0))

            # Extract features for ML model
            features = self._extract_ranking_features(result, user_profile, request)

            # Apply simple linear model (in production, use trained models)
            ml_score = self._apply_ranking_model(features, user_profile)

            # Combine with original score
            final_score = (
                original_score * 0.6 + ml_score * 0.4
            ) * request.personalization_strength

            ranking_factors = {
                "original_relevance": original_score,
                "ml_prediction": ml_score,
                "feature_count": len(features),
            }

            ranked_result = RankedResult(
                result_id=result["id"],
                title=result["title"],
                content=result.get("content", ""),
                original_score=original_score,
                personalized_score=ml_score,
                final_score=final_score,
                ranking_factors=ranking_factors,
                personalization_boost=ml_score - original_score,
                metadata=result,
            )

            ranked_results.append(ranked_result)

        ranked_results.sort(key=lambda r: r.final_score, reverse=True)
        return ranked_results

    async def _hybrid_ranking(
        self,
        results: list[dict[str, Any]],
        user_profile: UserProfile,
        request: PersonalizedRankingRequest,
    ) -> list[RankedResult]:
        """Apply hybrid ranking combining multiple strategies."""
        ranked_results = []

        for result in results:
            original_score = float(result.get("score", 0.0))

            # Calculate boosts from different strategies
            content_boost = (
                self._calculate_content_preference_boost(result, user_profile) * 0.3
            )

            behavioral_boost = (
                self._calculate_behavioral_boost(result, user_profile, request.context)
                * 0.3
            )

            contextual_boost = (
                self._calculate_contextual_boost(result, user_profile, request.context)
                * 0.2
            )

            # Add diversity and freshness factors
            diversity_boost = self._calculate_diversity_boost(result, results) * 0.1
            freshness_boost = self._calculate_freshness_boost(result) * 0.1

            # Combine all boosts
            total_boost = (
                content_boost
                + behavioral_boost
                + contextual_boost
                + diversity_boost
                + freshness_boost
            )

            personalization_boost = total_boost * request.personalization_strength
            final_score = min(1.0, original_score + personalization_boost)

            ranking_factors = {
                "original_relevance": original_score,
                "content_preference": content_boost,
                "behavioral_pattern": behavioral_boost,
                "contextual_relevance": contextual_boost,
                "diversity_factor": diversity_boost,
                "freshness_factor": freshness_boost,
            }

            ranked_result = RankedResult(
                result_id=result["id"],
                title=result["title"],
                content=result.get("content", ""),
                original_score=original_score,
                personalized_score=original_score + total_boost,
                final_score=final_score,
                ranking_factors=ranking_factors,
                personalization_boost=personalization_boost,
                metadata=result,
                ranking_explanation=(
                    self._generate_hybrid_explanation(ranking_factors)
                    if request.enable_explanations
                    else None
                ),
            )

            ranked_results.append(ranked_result)

        ranked_results.sort(key=lambda r: r.final_score, reverse=True)
        return ranked_results

    def _apply_default_ranking(
        self, results: list[dict[str, Any]]
    ) -> list[RankedResult]:
        """Apply non-personalized default ranking."""
        ranked_results = []

        for result in results:
            original_score = float(result.get("score", 0.0))

            ranked_result = RankedResult(
                result_id=result["id"],
                title=result["title"],
                content=result.get("content", ""),
                original_score=original_score,
                personalized_score=original_score,
                final_score=original_score,
                ranking_factors={"original_relevance": original_score},
                personalization_boost=0.0,
                metadata=result,
            )

            ranked_results.append(ranked_result)

        # Sort by original score
        ranked_results.sort(key=lambda r: r.original_score, reverse=True)
        return ranked_results

    def _calculate_content_preference_boost(
        self, result: dict[str, Any], user_profile: UserProfile
    ) -> float:
        """Calculate boost based on content preferences."""
        boost = 0.0

        # Check content type preferences
        content_type = result.get("content_type", "")
        if content_type in user_profile.preferred_result_types:
            boost += user_profile.preferred_result_types[content_type] * 0.3

        # Check specific attribute preferences
        for preference in user_profile.preferences:
            if preference.attribute in result:
                result_value = result[preference.attribute]
                if str(result_value).lower() == str(preference.value).lower():
                    boost += preference.weight * preference.confidence * 0.2

        return min(0.5, boost)  # Cap boost at 0.5

    def _calculate_behavioral_boost(
        self,
        result: dict[str, Any],
        user_profile: UserProfile,
        _context: dict[str, Any],
    ) -> float:
        """Calculate boost based on behavioral patterns."""
        boost = 0.0

        # Time-based preferences
        current_hour = datetime.now(tz=UTC).hour
        if current_hour in user_profile.active_hours:
            time_preference = user_profile.active_hours[current_hour]
            boost += time_preference * 0.1

        # Query pattern matching
        if "query_keywords" in result:
            keywords = result["query_keywords"]
            for keyword in keywords:
                if keyword in user_profile.query_patterns:
                    frequency = user_profile.query_patterns[keyword]
                    boost += min(0.1, frequency / 100.0)  # Normalize frequency

        # Exploration vs. exploitation
        if (
            user_profile.exploration_tendency > 0.7
            and result.get("novelty_score", 0) > 0.8
        ):
            # User likes exploring new content
            boost += 0.2

        return min(0.4, boost)

    def _calculate_contextual_boost(
        self,
        result: dict[str, Any],
        _user_profile: UserProfile,
        context: dict[str, Any],
    ) -> float:
        """Calculate boost based on current context."""
        boost = 0.0

        # Session context
        if (
            "domain" in context
            and "domain" in result
            and context["domain"] == result["domain"]
        ):
            boost += 0.2

        # Temporal context
        if result.get("time_sensitive"):
            recency = result.get("recency_score", 0)
            boost += recency * 0.3

        # Device/platform context
        if (
            "platform" in context
            and "platform_optimized" in result
            and context["platform"] in result["platform_optimized"]
        ):
            boost += 0.1

        return min(0.3, boost)

    def _calculate_diversity_boost(
        self, result: dict[str, Any], all_results: list[dict[str, Any]]
    ) -> float:
        """Calculate diversity boost to promote result variety."""
        # Simple diversity calculation based on content type
        content_type = result.get("content_type", "unknown")

        # Count how many results of the same type appear in top positions
        same_type_count = sum(
            1
            for r in all_results[:5]  # Check top 5
            if r.get("content_type") == content_type
        )

        # Boost diverse content types
        if same_type_count <= 1:
            return 0.1
        if same_type_count == 2:
            return 0.05
        return -0.05  # Slight penalty for over-represented types

    def _calculate_freshness_boost(self, result: dict[str, Any]) -> float:
        """Calculate freshness boost based on content age."""
        if "published_date" not in result:
            return 0.0

        try:
            published_date = datetime.fromisoformat(result["published_date"])
            age_days = (datetime.now(tz=UTC) - published_date).days

            # Boost fresher content
            if age_days <= 7:
                return 0.1
            if age_days <= 30:
                return 0.05
            if age_days <= 90:
                return 0.02
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _find_similar_users(self, user_id: str) -> list[str]:
        """Find users with similar preferences (simplified implementation)."""
        # In production, this would use collaborative filtering algorithms
        # For now, return a simple mock
        similar_users = []

        if user_id in self.user_similarity_matrix:
            similarities = self.user_similarity_matrix[user_id]
            # Get top 5 similar users
            similar_users = sorted(
                similarities.items(), key=lambda x: x[1], reverse=True
            )[:5]
            similar_users = [user for user, _ in similar_users]

        return similar_users

    def _calculate_collaborative_boost(
        self, result: dict[str, Any], similar_users: list[str]
    ) -> float:
        """Calculate collaborative filtering boost."""
        if not similar_users:
            return 0.0

        # Simple collaborative boost calculation
        # In production, would use interaction data from similar users
        boost = 0.0

        result_id = result["id"]
        positive_signals = 0

        for user in similar_users:
            if user in self.interaction_history:
                interactions = self.interaction_history[user]
                for interaction in interactions:
                    if (
                        interaction.result_id == result_id
                        and interaction.interaction_type
                        in [
                            InteractionType.CLICK,
                            InteractionType.BOOKMARK,
                            InteractionType.SHARE,
                        ]
                    ):
                        positive_signals += 1

        if positive_signals > 0:
            boost = min(0.3, positive_signals / len(similar_users))

        return boost

    def _extract_ranking_features(
        self,
        result: dict[str, Any],
        user_profile: UserProfile,
        request: PersonalizedRankingRequest,
    ) -> dict[str, float]:
        """Extract features for ML ranking model."""
        features = {
            "original_score": float(result.get("score", 0.0)),
            "user_total_interactions": float(user_profile.total_interactions),
            "user_profile_confidence": user_profile.confidence_score,
            "user_exploration_tendency": user_profile.exploration_tendency,
            "user_quality_sensitivity": user_profile.quality_sensitivity,
            "content_length": float(len(result.get("content", ""))),
            "title_length": float(len(result.get("title", ""))),
            "has_code_examples": float(bool(result.get("has_code", False))),
            "query_length": float(len(request.query.split())),
            "personalization_strength": request.personalization_strength,
        }

        # Add content type features
        content_type = result.get("content_type", "unknown")
        if content_type in user_profile.preferred_result_types:
            features["content_type_preference"] = user_profile.preferred_result_types[
                content_type
            ]
        else:
            features["content_type_preference"] = 0.0

        return features

    def _apply_ranking_model(
        self, features: dict[str, float], _user_profile: UserProfile
    ) -> float:
        """Apply ML ranking model (simplified linear model)."""
        # Simplified linear model weights
        weights = {
            "original_score": 0.4,
            "user_profile_confidence": 0.2,
            "content_type_preference": 0.2,
            "user_exploration_tendency": 0.1,
            "has_code_examples": 0.1,
        }

        score = 0.0
        for feature, value in features.items():
            if feature in weights:
                score += weights[feature] * value

        return min(1.0, max(0.0, score))

    def _get_time_preference_boost(self, user_profile: UserProfile) -> float:
        """Get boost based on time-of-day preferences."""
        current_hour = datetime.now(tz=UTC).hour
        if current_hour in user_profile.active_hours:
            return user_profile.active_hours[current_hour] * 0.1
        return 0.0

    def _get_query_pattern_boost(self, query: str, user_profile: UserProfile) -> float:
        """Get boost based on query pattern matching."""
        boost = 0.0
        query_words = query.lower().split()

        for word in query_words:
            if word in user_profile.query_patterns:
                frequency = user_profile.query_patterns[word]
                boost += min(0.05, frequency / 100.0)

        return boost

    def _get_session_context_boost(self, context: dict[str, Any]) -> float:
        """Get boost based on session context."""
        # Simple session context scoring
        boost = 0.0

        if context.get("session_length", 0) > 300:  # Long session
            boost += 0.1

        if context.get("previous_searches", 0) > 3:  # Active session
            boost += 0.05

        return boost

    def _get_temporal_context_boost(self, _context: dict[str, Any]) -> float:
        """Get boost based on temporal context."""
        boost = 0.0

        # Time of day
        hour = datetime.now(tz=UTC).hour
        if 9 <= hour <= 17:  # Business hours
            boost += 0.05
        elif 18 <= hour <= 22:  # Evening
            boost += 0.03

        # Day of week
        weekday = datetime.now(tz=UTC).weekday()
        if weekday < 5:  # Weekday
            boost += 0.02

        return boost

    async def _update_user_profile_from_interaction(
        self, interaction: InteractionEvent
    ) -> None:
        """Update user profile based on interaction."""
        user_id = interaction.user_id

        if user_id not in self.user_profiles:
            await self._get_user_profile(user_id)

        profile = self.user_profiles[user_id]

        # Update interaction count
        profile.total_interactions += 1

        # Update preferences based on interaction type
        if interaction.interaction_type in [
            InteractionType.CLICK,
            InteractionType.BOOKMARK,
            InteractionType.RATING,
        ]:
            # Positive signal - strengthen preferences
            self._strengthen_preferences(profile, interaction)
        elif interaction.interaction_type == InteractionType.SKIP:
            # Negative signal - weaken preferences
            self._weaken_preferences(profile, interaction)

        # Update temporal patterns
        hour = interaction.timestamp.hour
        if hour not in profile.active_hours:
            profile.active_hours[hour] = 0.0
        profile.active_hours[hour] = min(1.0, profile.active_hours[hour] + 0.1)

        # Update query patterns
        if interaction.query:
            words = interaction.query.lower().split()
            for word in words:
                if word not in profile.query_patterns:
                    profile.query_patterns[word] = 0
                profile.query_patterns[word] += 1

        # Update confidence score
        profile.confidence_score = min(1.0, profile.total_interactions / 50.0)

        profile.last_updated = datetime.now(tz=UTC)

    async def _build_profile_from_interactions(
        self, _profile: UserProfile, user_id: str
    ) -> None:
        """Build user profile from historical interactions."""
        if user_id not in self.interaction_history:
            return

        interactions = self.interaction_history[user_id]

        for interaction in interactions:
            # Extract preferences from interactions
            await self._update_user_profile_from_interaction(interaction)

    def _strengthen_preferences(
        self, profile: UserProfile, interaction: InteractionEvent
    ) -> None:
        """Strengthen user preferences based on positive interaction."""
        # This is a simplified implementation
        # In production, would extract more sophisticated preferences

    def _weaken_preferences(
        self, profile: UserProfile, interaction: InteractionEvent
    ) -> None:
        """Weaken user preferences based on negative interaction."""
        # This is a simplified implementation

    def _calculate_diversity_score(self, results: list[RankedResult]) -> float:
        """Calculate diversity score for ranked results."""
        if len(results) <= 1:
            return 1.0

        # Simple diversity calculation based on content types
        content_types = set()
        for result in results[:10]:  # Check top 10
            content_type = result.metadata.get("content_type", "unknown")
            content_types.add(content_type)

        # Normalize by maximum possible diversity
        max_diversity = min(10, len(results))
        diversity_score = len(content_types) / max_diversity

        return diversity_score

    def _calculate_coverage_score(
        self, results: list[RankedResult], user_profile: UserProfile
    ) -> float:
        """Calculate how well results cover user preferences."""
        if not user_profile.preferences:
            return 0.0

        covered_preferences = 0

        for preference in user_profile.preferences:
            for result in results[:10]:  # Check top 10
                if preference.attribute in result.metadata:
                    result_value = result.metadata[preference.attribute]
                    if str(result_value).lower() == str(preference.value).lower():
                        covered_preferences += 1
                        break

        coverage_score = covered_preferences / len(user_profile.preferences)
        return coverage_score

    def _calculate_reranking_impact(
        self, original_results: list[dict[str, Any]], ranked_results: list[RankedResult]
    ) -> float:
        """Calculate average position change from reranking."""
        if len(original_results) != len(ranked_results):
            return 0.0

        total_position_change = 0
        original_order = {result["id"]: i for i, result in enumerate(original_results)}

        for new_pos, ranked_result in enumerate(ranked_results):
            original_pos = original_order.get(ranked_result.result_id, new_pos)
            position_change = abs(new_pos - original_pos)
            total_position_change += position_change

        avg_position_change = total_position_change / len(ranked_results)
        return avg_position_change

    def _generate_content_explanation(
        self, content_boost: float, _user_profile: UserProfile
    ) -> str:
        """Generate explanation for content-based ranking."""
        if content_boost > 0.2:
            return "Ranked higher based on your content preferences"
        if content_boost > 0.1:
            return "Slightly boosted to match your interests"
        return "Standard ranking applied"

    def _generate_hybrid_explanation(self, ranking_factors: dict[str, float]) -> str:
        """Generate explanation for hybrid ranking."""
        primary_factor = max(ranking_factors.items(), key=lambda x: x[1])

        explanations = {
            "content_preference": "Matches your content interests",
            "behavioral_pattern": "Aligns with your usage patterns",
            "contextual_relevance": "Relevant to your current context",
            "diversity_factor": "Promotes result variety",
            "freshness_factor": "Recent and up-to-date content",
        }

        return explanations.get(primary_factor[0], "Personalized ranking applied")

    def _get_applied_factors(self, strategy: RankingStrategy) -> list[str]:
        """Get list of factors applied for a strategy."""
        factor_map = {
            RankingStrategy.CONTENT_BASED: ["content_preference", "attribute_matching"],
            RankingStrategy.COLLABORATIVE_FILTERING: [
                "user_similarity",
                "interaction_patterns",
            ],
            RankingStrategy.BEHAVIORAL: [
                "temporal_patterns",
                "query_patterns",
                "exploration_tendency",
            ],
            RankingStrategy.CONTEXTUAL: [
                "session_context",
                "temporal_context",
                "platform_context",
            ],
            RankingStrategy.LEARNING_TO_RANK: ["ml_features", "trained_model"],
            RankingStrategy.HYBRID: [
                "content_preference",
                "behavioral_patterns",
                "contextual_relevance",
                "diversity",
                "freshness",
            ],
            RankingStrategy.DEFAULT: ["original_relevance"],
        }

        return factor_map.get(strategy, [])

    def _update_performance_stats(
        self, strategy: RankingStrategy, processing_time: float, personalized: bool
    ) -> None:
        """Update performance statistics."""
        self.performance_stats["total_rankings"] += 1

        # Update average processing time
        total = self.performance_stats["total_rankings"]
        current_avg = self.performance_stats["avg_processing_time"]
        self.performance_stats["avg_processing_time"] = (
            current_avg * (total - 1) + processing_time
        ) / total

        # Update personalization rate
        if personalized:
            current_rate = self.performance_stats["personalization_rate"]
            self.performance_stats["personalization_rate"] = (
                current_rate * (total - 1) + 1.0
            ) / total
        else:
            current_rate = self.performance_stats["personalization_rate"]
            self.performance_stats["personalization_rate"] = (
                current_rate * (total - 1) + 0.0
            ) / total

        # Update strategy usage
        strategy_key = strategy.value
        if strategy_key not in self.performance_stats["strategy_usage"]:
            self.performance_stats["strategy_usage"][strategy_key] = 0
        self.performance_stats["strategy_usage"][strategy_key] += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.performance_stats,
            "cached_profiles": len(self.user_profiles),
            "users_with_history": len(self.interaction_history),
        }

    def clear_user_data(self, user_id: str) -> None:
        """Clear all data for a specific user."""
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
        if user_id in self.interaction_history:
            del self.interaction_history[user_id]
        if user_id in self.user_similarity_matrix:
            del self.user_similarity_matrix[user_id]