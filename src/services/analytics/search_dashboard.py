"""Search Analytics Dashboard for Query Insights.

This module provides an analytics dashboard for tracking search patterns,
performance metrics, user behavior, and optimization opportunities.
Showcasing data analytics and visualization capabilities.
"""

import asyncio

# json import removed (unused)
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

# get_config import removed (unused)
from src.services.base import BaseService


logger = logging.getLogger(__name__)


class QueryPattern(BaseModel):
    """Represents a discovered query pattern."""

    pattern: str = Field(..., description="Query pattern template")
    frequency: int = Field(..., description="Number of occurrences")
    avg_performance: float = Field(..., description="Average processing time")
    success_rate: float = Field(..., description="Success rate percentage")
    sample_queries: list[str] = Field(
        ..., description="Sample queries matching pattern"
    )


class PerformanceMetric(BaseModel):
    """Performance metric data point."""

    timestamp: datetime = Field(..., description="Metric timestamp")
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Metric value")
    tags: dict[str, str] = Field(default_factory=dict, description="Metric tags")


class UserBehaviorInsight(BaseModel):
    """User behavior analysis insight."""

    insight_type: str = Field(..., description="Type of insight")
    description: str = Field(..., description="Human-readable description")
    impact_score: float = Field(..., description="Impact score 0-1")
    recommendations: list[str] = Field(..., description="Actionable recommendations")
    data_points: dict[str, Any] = Field(..., description="Supporting data")


class SearchAnalyticsDashboard(BaseService):
    """Search analytics dashboard for query patterns and performance insights.

    This demonstrates:
    - Data analytics and pattern recognition
    - Performance monitoring and optimization insights
    - User behavior analysis and recommendation systems
    - Data visualization preparation and metrics aggregation
    - Capabilities for search systems
    """

    def __init__(self):
        """Initialize the search analytics dashboard."""
        super().__init__()
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Analytics data storage (in-memory for demo, would use proper DB in production)
        self.query_history: list[dict[str, Any]] = []
        self.performance_metrics: list[PerformanceMetric] = []
        self.user_patterns: dict[str, list[dict[str, Any]]] = {}

        # Pattern detection state
        self.detected_patterns: list[QueryPattern] = []
        self.pattern_detection_interval = 300  # 5 minutes
        self.last_pattern_analysis = 0

        # Metrics
        self.realtime_stats = {
            "queries_last_hour": 0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0,
            "active_users": 0,
            "popular_queries": [],
            "performance_alerts": [],
        }

    async def initialize(self) -> None:
        """Initialize the analytics dashboard."""
        if self._initialized:
            return

        try:
            # Initialize background pattern detection
            pattern_task = asyncio.create_task(self._pattern_detection_loop())
            # Store reference to prevent task garbage collection
            pattern_task.add_done_callback(
                lambda _: self._logger.debug("Pattern detection loop completed")
            )

            self._initialized = True
            self._logger.info("SearchAnalyticsDashboard initialized successfully")

        except (AttributeError, ImportError, OSError):
            self._logger.exception("Failed to initialize SearchAnalyticsDashboard")
            raise

    async def track_query(
        self,
        query: str,
        user_id: str | None = None,
        processing_time_ms: float = 0.0,
        success: bool = True,
        features_used: list[str] | None = None,
        results_count: int = 0,
        cache_hit: bool = False,
        **metadata: Any,
    ) -> None:
        """Track a search query for analytics.

        Args:
            query: The search query
            user_id: Optional user identifier
            processing_time_ms: Query processing time
            success: Whether query was successful
            features_used: List of features used in processing
            results_count: Number of results returned
            cache_hit: Whether result was from cache
            **metadata: Additional metadata to track

        """
        try:
            query_data = {
                "query": query,
                "user_id": user_id or "anonymous",
                "timestamp": datetime.now(tz=UTC),
                "processing_time_ms": processing_time_ms,
                "success": success,
                "features_used": features_used or [],
                "results_count": results_count,
                "cache_hit": cache_hit,
                "query_length": len(query),
                "word_count": len(query.split()),
                **metadata,
            }

            # Store query data
            self.query_history.append(query_data)

            # Update user patterns
            if user_id:
                if user_id not in self.user_patterns:
                    self.user_patterns[user_id] = []
                self.user_patterns[user_id].append(query_data)

                # Keep only last 100 queries per user
                if len(self.user_patterns[user_id]) > 100:
                    self.user_patterns[user_id] = self.user_patterns[user_id][-100:]

            # Keep only last 1000 queries total
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]

            # Update stats
            await self._update_realtime_stats()

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to track query")

    async def track_performance_metric(
        self, metric_name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Track a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for categorization

        """
        try:
            metric = PerformanceMetric(
                timestamp=datetime.now(tz=UTC),
                metric_name=metric_name,
                value=value,
                tags=tags or {},
            )

            self.performance_metrics.append(metric)

            # Keep only last 500 metrics
            if len(self.performance_metrics) > 500:
                self.performance_metrics = self.performance_metrics[-500:]

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to track performance metric")

    async def get_realtime_dashboard(self) -> dict[str, Any]:
        """Get real-time dashboard data.

        Returns:
            Dashboard data including metrics, patterns, and insights

        """
        try:
            # Ensure patterns are up to date
            await self._detect_query_patterns()

            # Generate user behavior insights
            insights = await self._generate_behavior_insights()

            # Get performance trends
            trends = await self._calculate_performance_trends()

            return {
                "realtime_stats": self.realtime_stats.copy(),
                "query_patterns": [
                    pattern.model_dump() for pattern in self.detected_patterns[:10]
                ],
                "performance_trends": trends,
                "user_insights": [insight.model_dump() for insight in insights[:5]],
                "feature_utilization": await self._calculate_feature_utilization(),
                "query_volume_timeline": await self._get_query_volume_timeline(),
                "top_performing_queries": await self._get_top_performing_queries(),
                "optimization_opportunities": (
                    await self._identify_optimization_opportunities()
                ),
                "last_updated": datetime.now(tz=UTC).isoformat(),
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to generate realtime dashboard")
            return {"error": "Failed to generate dashboard data"}

    async def get_query_analytics(
        self, time_range_hours: int = 24, user_id: str | None = None
    ) -> dict[str, Any]:
        """Get detailed query analytics for a time range.

        Args:
            time_range_hours: Hours to look back
            user_id: Optional specific user to analyze

        Returns:
            Detailed analytics data

        """
        try:
            cutoff_time = datetime.now(tz=UTC) - timedelta(hours=time_range_hours)

            # Filter queries by time range and user
            filtered_queries = [
                q
                for q in self.query_history
                if q["timestamp"] >= cutoff_time
                and (user_id is None or q["user_id"] == user_id)
            ]

            if not filtered_queries:
                return {"message": "No queries found for the specified criteria"}

            # Calculate analytics
            total_queries = len(filtered_queries)
            successful_queries = sum(1 for q in filtered_queries if q["success"])
            cache_hits = sum(1 for q in filtered_queries if q["cache_hit"])
            avg_processing_time = (
                sum(q["processing_time_ms"] for q in filtered_queries) / total_queries
            )

            # Query complexity analysis
            query_lengths = [q["query_length"] for q in filtered_queries]
            word_counts = [q["word_count"] for q in filtered_queries]

            # Feature usage analysis
            all_features = []
            for q in filtered_queries:
                all_features.extend(q.get("features_used", []))

            feature_usage = {}
            for feature in set(all_features):
                feature_usage[feature] = all_features.count(feature)

            return {
                "time_range_hours": time_range_hours,
                "user_id": user_id,
                "total_queries": total_queries,
                "success_rate": successful_queries / total_queries
                if total_queries > 0
                else 0,
                "cache_hit_rate": cache_hits / total_queries
                if total_queries > 0
                else 0,
                "avg_processing_time_ms": avg_processing_time,
                "query_complexity": {
                    "avg_length": sum(query_lengths) / len(query_lengths)
                    if query_lengths
                    else 0,
                    "avg_word_count": sum(word_counts) / len(word_counts)
                    if word_counts
                    else 0,
                    "length_distribution": self._calculate_distribution(query_lengths),
                },
                "feature_utilization": feature_usage,
                "temporal_patterns": await self._analyze_temporal_patterns(
                    filtered_queries
                ),
                "performance_distribution": self._calculate_distribution(
                    [q["processing_time_ms"] for q in filtered_queries]
                ),
                "generated_at": datetime.now(tz=UTC).isoformat(),
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to generate query analytics")
            return {"error": "Failed to generate analytics"}

    async def get_optimization_recommendations(self) -> list[dict[str, Any]]:
        """Get optimization recommendations based on analytics.

        Returns:
            List of actionable optimization recommendations

        """
        try:
            recommendations = []

            # Analyze recent performance
            recent_queries = [
                q
                for q in self.query_history
                if q["timestamp"] >= datetime.now(tz=UTC) - timedelta(hours=1)
            ]

            if recent_queries:
                avg_time = sum(q["processing_time_ms"] for q in recent_queries) / len(
                    recent_queries
                )
                cache_rate = sum(1 for q in recent_queries if q["cache_hit"]) / len(
                    recent_queries
                )

                # Performance recommendations
                if avg_time > 1000:  # >1 second
                    recommendations.append(
                        {
                            "type": "performance",
                            "priority": "high",
                            "title": "High Query Processing Time",
                            "description": (
                                f"Average processing time is {avg_time:.0f}ms, "
                                "consider optimization"
                            ),
                            "actions": [
                                "Enable result caching for frequent queries",
                                "Optimize vector index parameters",
                                "Consider query preprocessing improvements",
                            ],
                            "impact": "Could reduce response time by 30-50%",
                        }
                    )

                # Cache recommendations
                if cache_rate < 0.3:  # <30% cache hits
                    recommendations.append(
                        {
                            "type": "caching",
                            "priority": "medium",
                            "title": "Low Cache Hit Rate",
                            "description": (
                                f"Cache hit rate is {cache_rate:.1%}, "
                                "caching strategy could be improved"
                            ),
                            "actions": [
                                "Increase cache size",
                                "Improve cache key generation",
                                "Consider query normalization for better cache hits",
                            ],
                            "impact": "Could improve cache hit rate to 60-80%",
                        }
                    )

            # Pattern-based recommendations
            if self.detected_patterns:
                frequent_patterns = [
                    p for p in self.detected_patterns if p.frequency > 10
                ]
                if frequent_patterns:
                    recommendations.append(
                        {
                            "type": "patterns",
                            "priority": "medium",
                            "title": "Frequent Query Patterns Detected",
                            "description": (
                                f"Found {len(frequent_patterns)} "
                                "frequent query patterns"
                            ),
                            "actions": [
                                "Create optimized search templates for common patterns",
                                "Pre-compute results for frequent queries",
                                "Consider auto-completion based on patterns",
                            ],
                            "impact": (
                                "Could improve user experience and reduce server load"
                            ),
                        }
                    )

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to generate optimization recommendations")
            return []

        return recommendations

    async def _update_realtime_stats(self) -> None:
        """Update real-time statistics."""
        try:
            # Get queries from last hour
            hour_ago = datetime.now(tz=UTC) - timedelta(hours=1)
            recent_queries = [
                q for q in self.query_history if q["timestamp"] >= hour_ago
            ]

            if recent_queries:
                self.realtime_stats.update(
                    {
                        "queries_last_hour": len(recent_queries),
                        "avg_response_time": sum(
                            q["processing_time_ms"] for q in recent_queries
                        )
                        / len(recent_queries),
                        "success_rate": sum(1 for q in recent_queries if q["success"])
                        / len(recent_queries),
                        "cache_hit_rate": sum(
                            1 for q in recent_queries if q["cache_hit"]
                        )
                        / len(recent_queries),
                        "error_rate": sum(1 for q in recent_queries if not q["success"])
                        / len(recent_queries),
                        "active_users": len({q["user_id"] for q in recent_queries}),
                    }
                )

                # Update popular queries
                query_counts = {}
                for q in recent_queries:
                    query_text = q["query"].lower()
                    query_counts[query_text] = query_counts.get(query_text, 0) + 1

                popular = sorted(
                    query_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
                self.realtime_stats["popular_queries"] = [
                    {"query": query, "count": count} for query, count in popular
                ]

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to update realtime stats")

    async def _detect_query_patterns(self) -> None:
        """Detect common query patterns."""
        try:
            current_time = time.time()
            if (
                current_time - self.last_pattern_analysis
                < self.pattern_detection_interval
            ):
                return

            # Simple pattern detection based on query structure
            patterns = {}

            for query_data in self.query_history[-200:]:  # Analyze last 200 queries
                query = query_data["query"].lower()

                # Create pattern by replacing specific terms with placeholders
                pattern = self._extract_pattern(query)

                if pattern not in patterns:
                    patterns[pattern] = {
                        "queries": [],
                        "performance_times": [],
                        "success_count": 0,
                    }

                patterns[pattern]["queries"].append(query)
                patterns[pattern]["performance_times"].append(
                    query_data["processing_time_ms"]
                )
                if query_data["success"]:
                    patterns[pattern]["success_count"] += 1

            # Convert to QueryPattern objects
            self.detected_patterns = []
            for pattern, data in patterns.items():
                if len(data["queries"]) >= 3:  # At least 3 occurrences
                    query_pattern = QueryPattern(
                        pattern=pattern,
                        frequency=len(data["queries"]),
                        avg_performance=sum(data["performance_times"])
                        / len(data["performance_times"]),
                        success_rate=data["success_count"] / len(data["queries"]),
                        sample_queries=data["queries"][:3],
                    )
                    self.detected_patterns.append(query_pattern)

            # Sort by frequency
            self.detected_patterns.sort(key=lambda x: x.frequency, reverse=True)
            self.last_pattern_analysis = current_time

        except (OSError, PermissionError):
            self._logger.exception("Failed to detect query patterns")

    def _extract_pattern(self, query: str) -> str:
        """Extract a pattern from a query by replacing specific terms."""
        # Simple pattern extraction - replace numbers and specific terms
        # Replace numbers
        pattern = re.sub(r"\d+", "[NUMBER]", query)

        # Replace common variable terms
        replacements = [
            (r"\b\w+\.(py|js|html|css|json)\b", "[FILENAME]"),
            (r"\bhttps?://\S+", "[URL]"),
            (r"\b\w+@\w+\.\w+", "[EMAIL]"),
            (r"\b[A-Z][a-z]+[A-Z]\w*", "[CAMELCASE]"),
        ]

        for regex, replacement in replacements:
            pattern = re.sub(regex, replacement, pattern)

        return pattern

    async def _generate_behavior_insights(self) -> list[UserBehaviorInsight]:
        """Generate user behavior insights."""
        try:
            insights = []

            # Analyze query complexity trends
            if len(self.query_history) >= 10:
                recent_lengths = [q["query_length"] for q in self.query_history[-20:]]
                older_lengths = [q["query_length"] for q in self.query_history[-40:-20]]

                if older_lengths:
                    recent_avg = sum(recent_lengths) / len(recent_lengths)
                    older_avg = sum(older_lengths) / len(older_lengths)

                    if recent_avg > older_avg * 1.2:  # 20% increase
                        insights.append(
                            UserBehaviorInsight(
                                insight_type="query_complexity",
                                description=(
                                    "Users are asking increasingly complex questions"
                                ),
                                impact_score=0.7,
                                recommendations=[
                                    "Consider adding query assistance features",
                                    "Implement query reformulation suggestions",
                                    "Add examples of effective query patterns",
                                ],
                                data_points={
                                    "recent_avg_length": recent_avg,
                                    "previous_avg_length": older_avg,
                                    "increase_percentage": (
                                        (recent_avg - older_avg) / older_avg * 100
                                    ),
                                },
                            )
                        )

            # Analyze feature usage patterns
            all_features = []
            for q in self.query_history[-50:]:
                all_features.extend(q.get("features_used", []))

            if all_features:
                feature_counts = {}
                for feature in all_features:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1

                most_used = max(feature_counts.items(), key=lambda x: x[1])
                if most_used[1] > len(self.query_history[-50:]) * 0.8:  # >80% usage
                    insights.append(
                        UserBehaviorInsight(
                            insight_type="feature_adoption",
                            description=f"High adoption of '{most_used[0]}' feature",
                            impact_score=0.6,
                            recommendations=[
                                f"Optimize {most_used[0]} performance "
                                "for better user experience",
                                "Consider making this feature more prominent",
                                "Explore similar features users might find valuable",
                            ],
                            data_points={
                                "feature": most_used[0],
                                "usage_rate": most_used[1]
                                / len(self.query_history[-50:]),
                            },
                        )
                    )

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to generate behavior insights")
            return []

        return insights

    async def _calculate_performance_trends(self) -> dict[str, Any]:
        """Calculate performance trends over time."""
        try:
            if len(self.query_history) < 10:
                return {"message": "Insufficient data for trend analysis"}

            # Split into time buckets
            recent_queries = self.query_history[-20:]
            older_queries = (
                self.query_history[-40:-20] if len(self.query_history) >= 40 else []
            )

            trends = {}

            if older_queries:
                # Calculate trends
                recent_avg_time = sum(
                    q["processing_time_ms"] for q in recent_queries
                ) / len(recent_queries)
                older_avg_time = sum(
                    q["processing_time_ms"] for q in older_queries
                ) / len(older_queries)

                time_trend = (
                    "improving" if recent_avg_time < older_avg_time else "degrading"
                )

                recent_success = sum(1 for q in recent_queries if q["success"]) / len(
                    recent_queries
                )
                older_success = sum(1 for q in older_queries if q["success"]) / len(
                    older_queries
                )

                success_trend = (
                    "improving" if recent_success > older_success else "stable"
                )

                trends = {
                    "response_time": {
                        "trend": time_trend,
                        "recent_avg": recent_avg_time,
                        "previous_avg": older_avg_time,
                        "change_percent": (
                            (recent_avg_time - older_avg_time) / older_avg_time * 100
                        ),
                    },
                    "success_rate": {
                        "trend": success_trend,
                        "recent_rate": recent_success,
                        "previous_rate": older_success,
                        "change_percent": (
                            (recent_success - older_success) / older_success * 100
                            if older_success > 0
                            else 0
                        ),
                    },
                }

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to calculate performance trends")
            return {}

        return trends

    async def _calculate_feature_utilization(self) -> dict[str, float]:
        """Calculate feature utilization rates."""
        try:
            if not self.query_history:
                return {}

            feature_counts = {}
            total_queries = len(self.query_history[-100:])  # Last 100 queries

            for query_data in self.query_history[-100:]:
                for feature in query_data.get("features_used", []):
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1

            # Calculate utilization rates
            utilization = {}
            for feature, count in feature_counts.items():
                utilization[feature] = count / total_queries

        except (OSError, PermissionError):
            self._logger.exception("Failed to calculate feature utilization")
            return {}

        return utilization

    def _calculate_distribution(self, values: list[float]) -> dict[str, float]:
        """Calculate distribution statistics for a list of values."""
        if not values:
            return {}

        sorted_values = sorted(values)
        length = len(sorted_values)

        return {
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "median": sorted_values[length // 2],
            "p25": sorted_values[length // 4],
            "p75": sorted_values[3 * length // 4],
            "p95": sorted_values[int(0.95 * length)],
        }

    async def _get_query_volume_timeline(self) -> list[dict[str, Any]]:
        """Get query volume timeline for the last 24 hours."""
        try:
            # Group queries by hour
            hour_buckets = {}
            cutoff = datetime.now(tz=UTC) - timedelta(hours=24)

            for query_data in self.query_history:
                if query_data["timestamp"] >= cutoff:
                    hour = query_data["timestamp"].replace(
                        minute=0, second=0, microsecond=0
                    )
                    hour_key = hour.isoformat()

                    if hour_key not in hour_buckets:
                        hour_buckets[hour_key] = 0
                    hour_buckets[hour_key] += 1

            # Convert to timeline format
            timeline = []
            for hour_key, count in sorted(hour_buckets.items()):
                timeline.append({"timestamp": hour_key, "query_count": count})

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to get query volume timeline")
            return []

        return timeline

    async def _get_top_performing_queries(self) -> list[dict[str, Any]]:
        """Get top performing queries by success rate and speed."""
        try:
            query_performance = {}

            for query_data in self.query_history[-100:]:  # Last 100 queries
                query = query_data["query"]

                if query not in query_performance:
                    query_performance[query] = {
                        "query": query,
                        "executions": 0,
                        "successes": 0,
                        "total_time": 0.0,
                        "results_count": 0,
                    }

                perf = query_performance[query]
                perf["executions"] += 1
                if query_data["success"]:
                    perf["successes"] += 1
                perf["total_time"] += query_data["processing_time_ms"]
                perf["results_count"] += query_data.get("results_count", 0)

            # Calculate metrics and sort
            top_queries = []
            for query, perf in query_performance.items():
                if perf["executions"] >= 2:  # At least 2 executions
                    success_rate = perf["successes"] / perf["executions"]
                    avg_time = perf["total_time"] / perf["executions"]
                    avg_results = perf["results_count"] / perf["executions"]

                    # Simple performance score (higher is better)
                    score = success_rate * 100 - (avg_time / 100) + (avg_results / 10)

                    top_queries.append(
                        {
                            "query": query,
                            "performance_score": score,
                            "success_rate": success_rate,
                            "avg_response_time": avg_time,
                            "avg_results": avg_results,
                            "executions": perf["executions"],
                        }
                    )

            # Sort by performance score and return top 10
            top_queries.sort(key=lambda x: x["performance_score"], reverse=True)
            return top_queries[:10]

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to get top performing queries")
            return []

    async def _identify_optimization_opportunities(self) -> list[dict[str, Any]]:
        """Identify optimization opportunities based on analytics."""
        try:
            opportunities = []

            # Analyze slow queries
            slow_queries = [
                q
                for q in self.query_history[-100:]
                if q["processing_time_ms"] > 2000  # >2 seconds
            ]

            if slow_queries:
                opportunities.append(
                    {
                        "type": "performance",
                        "title": "Slow Query Optimization",
                        "description": f"{len(slow_queries)} queries taking >2 seconds",
                        "priority": "high",
                        "potential_improvement": "40-60% response time reduction",
                        "sample_queries": [q["query"] for q in slow_queries[:3]],
                    }
                )

            # Analyze failed queries
            failed_queries = [q for q in self.query_history[-100:] if not q["success"]]

            if len(failed_queries) > 5:
                opportunities.append(
                    {
                        "type": "reliability",
                        "title": "Error Rate Reduction",
                        "description": f"{len(failed_queries)} failed queries detected",
                        "priority": "medium",
                        "potential_improvement": "Improved user satisfaction",
                        "sample_queries": [q["query"] for q in failed_queries[:3]],
                    }
                )

            # Analyze cache opportunities
            cache_misses = [q for q in self.query_history[-100:] if not q["cache_hit"]]

            if len(cache_misses) > 70:  # >70% cache miss rate
                opportunities.append(
                    {
                        "type": "caching",
                        "title": "Cache Hit Rate Improvement",
                        "description": f"High cache miss rate: {len(cache_misses)}%",
                        "priority": "medium",
                        "potential_improvement": "30-50% response time improvement",
                        "recommendations": [
                            "Increase cache size",
                            "Improve cache key normalization",
                            "Pre-warm cache with popular queries",
                        ],
                    }
                )

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to identify optimization opportunities")
            return []

        return opportunities

    async def _pattern_detection_loop(self) -> None:
        """Background loop for pattern detection."""
        while self._initialized:
            try:
                await asyncio.sleep(self.pattern_detection_interval)
                await self._detect_query_patterns()
            except asyncio.CancelledError:
                break
            except (TimeoutError, OSError, PermissionError):
                self._logger.exception("Error in pattern detection loop")

    async def _analyze_temporal_patterns(
        self, queries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze temporal patterns in query data."""
        try:
            if not queries:
                return {}

            # Group by hour of day
            hour_counts = {}
            for query_data in queries:
                hour = query_data["timestamp"].hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

            # Find peak hours
            peak_hour = (
                max(hour_counts.items(), key=lambda x: x[1]) if hour_counts else (0, 0)
            )

            # Group by day of week
            weekday_counts = {}
            for query_data in queries:
                weekday = query_data["timestamp"].weekday()  # 0=Monday, 6=Sunday
                weekday_counts[weekday] = weekday_counts.get(weekday, 0) + 1

            weekday_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            weekday_distribution = {
                weekday_names[day]: count for day, count in weekday_counts.items()
            }

            return {
                "peak_hour": peak_hour[0],
                "peak_hour_count": peak_hour[1],
                "hourly_distribution": hour_counts,
                "weekday_distribution": weekday_distribution,
                "query_span_hours": (
                    (
                        max(q["timestamp"] for q in queries)
                        - min(q["timestamp"] for q in queries)
                    ).total_seconds()
                    / 3600
                    if len(queries) > 1
                    else 0
                ),
            }

        except (ConnectionError, OSError, PermissionError):
            self._logger.exception("Failed to analyze temporal patterns")
            return {}

    async def cleanup(self) -> None:
        """Cleanup dashboard resources."""
        if self._initialized:
            self._initialized = False
            self._logger.info("SearchAnalyticsDashboard cleaned up")
