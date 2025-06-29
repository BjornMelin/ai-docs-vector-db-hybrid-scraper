"""Focused QdrantIndexing module for payload indexing operations.

This module provides a clean, focused implementation of indexing functionality
extracted from QdrantService, focusing specifically on payload index management.
"""

import logging
import time
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.config import Config
from src.services.errors import QdrantServiceError


logger = logging.getLogger(__name__)


class QdrantIndexing:
    """Focused payload indexing operations for Qdrant with performance optimization."""

    def __init__(self, client: AsyncQdrantClient, config: Config):
        """Initialize indexing service.

        Args:
            client: Initialized Qdrant client
            config: Unified configuration

        """
        self.client = client
        self.config = config

    async def create_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes on key metadata fields for 10-100x faster filtering.

        Creates indexes on high-value fields for dramatic search performance improvements:
        - Keyword indexes for exact matching (site_name, embedding_model, etc.)
        - Text indexes for partial matching (title, content_preview)
        - Integer indexes for range queries (scraped_at, word_count, etc.)

        Args:
            collection_name: Collection to index

        Raises:
            QdrantServiceError: If index creation fails

        """
        try:
            logger.info(
                f"Creating payload indexes for collection: {collection_name}"
            )  # TODO: Convert f-string to logging format

            # Keyword indexes for exact matching (categorical data)
            keyword_fields = [
                # Core indexable fields per documentation
                "doc_type",  # "api", "guide", "tutorial", "reference"
                "language",  # "python", "typescript", "rust"
                "framework",  # "fastapi", "nextjs", "react"
                "version",  # "3.0", "14.2", "latest"
                "crawl_source",  # "crawl4ai", "browser_use", "playwright"
                # Additional system fields
                "site_name",  # Documentation site name
                "embedding_model",  # Embedding model used
                "embedding_provider",  # Provider (openai, fastembed)
                "search_strategy",  # Strategy (hybrid, dense, sparse)
                "scraper_version",  # Scraper version
            ]

            for field in keyword_fields:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
                logger.debug(
                    f"Created keyword index for field: {field}"
                )  # TODO: Convert f-string to logging format

            # Text indexes for partial matching (full-text search)
            text_fields = [
                "title",  # Document titles
                "content_preview",  # Content previews
            ]

            for field in text_fields:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.TEXT,
                    wait=True,
                )
                logger.debug(
                    f"Created text index for field: {field}"
                )  # TODO: Convert f-string to logging format

            # Integer indexes for range queries
            integer_fields = [
                # Core timestamp fields per documentation
                "created_at",  # Document creation timestamp
                "last_updated",  # Last update timestamp
                "crawl_timestamp",  # Modern crawling timestamp field
                # Content metrics
                "word_count",  # Content length filtering
                "char_count",  # Character count filtering
                "quality_score",  # Content quality score
                # Document structure
                "chunk_index",  # Chunk position filtering
                "total_chunks",  # Document size filtering
                "depth",  # Crawl depth filtering
                "links_count",  # Links count filtering
            ]

            for field in integer_fields:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.INTEGER,
                    wait=True,
                )
                logger.debug(
                    f"Created integer index for field: {field}"
                )  # TODO: Convert f-string to logging format

            logger.info(
                f"Successfully created {len(keyword_fields) + len(text_fields) + len(integer_fields)} "
                f"payload indexes for collection: {collection_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to create payload indexes for collection {collection_name}: {e}",
                exc_info=True,
            )
            msg = f"Failed to create payload indexes: {e}"
            raise QdrantServiceError(msg) from e

    async def list_payload_indexes(self, collection_name: str) -> list[str]:
        """List all payload indexes in a collection.

        Args:
            collection_name: Collection to check

        Returns:
            List of indexed field names

        Raises:
            QdrantServiceError: If listing fails

        """
        try:
            collection_info = await self.client.get_collection(collection_name)

            # Extract indexed fields from payload schema
            indexed_fields = []
            if (
                hasattr(collection_info, "payload_schema")
                and collection_info.payload_schema
            ):
                for field_name, field_info in collection_info.payload_schema.items():
                    # Check if field has index configuration
                    if hasattr(field_info, "index") and field_info.index:
                        indexed_fields.append(field_name)

            logger.info(
                f"Found {len(indexed_fields)} indexed fields in {collection_name}"
            )
            return indexed_fields

        except Exception as e:
            logger.error(
                f"Failed to list payload indexes for collection {collection_name}: {e}",
                exc_info=True,
            )
            msg = f"Failed to list payload indexes: {e}"
            raise QdrantServiceError(msg) from e

    async def drop_payload_index(self, collection_name: str, field_name: str) -> None:
        """Drop a specific payload index.

        Args:
            collection_name: Collection containing the index
            field_name: Field to drop index for

        Raises:
            QdrantServiceError: If drop fails

        """
        try:
            await self.client.delete_payload_index(
                collection_name=collection_name, field_name=field_name, wait=True
            )
            logger.info(
                f"Dropped payload index for field: {field_name}"
            )  # TODO: Convert f-string to logging format

        except Exception as e:
            logger.error(
                f"Failed to drop payload index for field {field_name}: {e}",
                exc_info=True,
            )
            msg = f"Failed to drop payload index: {e}"
            raise QdrantServiceError(msg) from e

    async def reindex_collection(self, collection_name: str) -> None:
        """Reindex all payload fields for a collection.

        Useful after bulk updates or when index performance degrades.

        Args:
            collection_name: Collection to reindex

        Raises:
            QdrantServiceError: If reindexing fails

        """
        try:
            logger.info(
                f"Starting full reindex for collection: {collection_name}"
            )  # TODO: Convert f-string to logging format

            # Get existing indexes
            existing_indexes = await self.list_payload_indexes(collection_name)

            # Drop existing indexes
            for field_name in existing_indexes:
                try:
                    await self.drop_payload_index(collection_name, field_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to drop index for {field_name}: {e}"
                    )  # TODO: Convert f-string to logging format

            # Recreate all indexes
            await self.create_payload_indexes(collection_name)

            logger.info(
                f"Successfully reindexed collection: {collection_name}"
            )  # TODO: Convert f-string to logging format

        except Exception as e:
            logger.error(
                f"Failed to reindex collection {collection_name}: {e}", exc_info=True
            )
            msg = f"Failed to reindex collection: {e}"
            raise QdrantServiceError(msg) from e

    async def get_payload_index_stats(self, collection_name: str) -> dict[str, Any]:
        """Get statistics about payload indexes in a collection.

        Args:
            collection_name: Collection to analyze

        Returns:
            Dictionary with index statistics

        Raises:
            QdrantServiceError: If stats retrieval fails

        """
        try:
            collection_info = await self.client.get_collection(collection_name)
            indexed_fields = await self.list_payload_indexes(collection_name)

            stats = {
                "collection_name": collection_name,
                "total_points": collection_info.points_count or 0,
                "indexed_fields_count": len(indexed_fields),
                "indexed_fields": indexed_fields,
                "payload_schema": {},
            }

            # Add payload schema information if available
            if (
                hasattr(collection_info, "payload_schema")
                and collection_info.payload_schema
            ):
                for field_name, field_info in collection_info.payload_schema.items():
                    stats["payload_schema"][field_name] = {
                        "indexed": field_name in indexed_fields,
                        "type": str(getattr(field_info, "data_type", "unknown")),
                    }

            return stats

        except Exception as e:
            logger.error(
                f"Failed to get payload index stats for collection {collection_name}: {e}",
                exc_info=True,
            )
            msg = f"Failed to get payload index stats: {e}"
            raise QdrantServiceError(msg) from e

    async def validate_index_health(self, collection_name: str) -> dict[str, Any]:
        """Validate the health and status of payload indexes for a collection.

        Args:
            collection_name: Collection to validate

        Returns:
            Health status report with validation results

        Raises:
            QdrantServiceError: If validation fails

        """
        try:
            logger.info(
                f"Validating payload index health for collection: {collection_name}"
            )

            # Get collection information
            collection_info = await self.client.get_collection(collection_name)
            indexed_fields = await self.list_payload_indexes(collection_name)

            # Define expected core indexes
            expected_keyword_fields = [
                "doc_type",
                "language",
                "framework",
                "version",
                "crawl_source",
                "site_name",
                "embedding_model",
                "embedding_provider",
            ]
            expected_text_fields = ["title", "content_preview"]
            expected_integer_fields = [
                "created_at",
                "last_updated",
                "word_count",
                "char_count",
            ]

            all_expected = (
                expected_keyword_fields + expected_text_fields + expected_integer_fields
            )

            # Check payload index status
            missing_indexes = [
                field for field in all_expected if field not in indexed_fields
            ]
            extra_indexes = [
                field for field in indexed_fields if field not in all_expected
            ]

            # Calculate health score
            expected_count = len(all_expected)
            present_count = len([f for f in all_expected if f in indexed_fields])
            health_score = (
                (present_count / expected_count) * 100 if expected_count > 0 else 0
            )

            # Determine health status
            if health_score >= 95:
                status = "healthy"
            elif health_score >= 80:
                status = "warning"
            else:
                status = "critical"

            health_report = {
                "collection_name": collection_name,
                "status": status,
                "health_score": round(health_score, 2),
                "total_points": collection_info.points_count or 0,
                "index_summary": {
                    "expected_indexes": expected_count,
                    "present_indexes": present_count,
                    "missing_indexes": len(missing_indexes),
                    "extra_indexes": len(extra_indexes),
                },
                "payload_indexes": {
                    "missing_indexes": missing_indexes,
                    "extra_indexes": extra_indexes,
                },
                "recommendations": self._generate_index_recommendations(
                    missing_indexes, extra_indexes, status
                ),
                "validation_timestamp": int(time.time()),
            }

            logger.info(
                f"Payload index health validation completed for {collection_name}: "
                f"Status={status}, Score={health_score:.1f}%"
            )

            return health_report

        except Exception as e:
            logger.error(
                f"Failed to validate index health for collection {collection_name}: {e}",
                exc_info=True,
            )
            msg = f"Failed to validate index health: {e}"
            raise QdrantServiceError(msg) from e

    async def get_index_usage_stats(self, collection_name: str) -> dict[str, Any]:
        """Get detailed usage statistics for payload indexes.

        Args:
            collection_name: Collection to analyze

        Returns:
            Detailed index usage statistics

        Raises:
            QdrantServiceError: If stats retrieval fails

        """
        try:
            logger.debug(
                f"Collecting index usage stats for collection: {collection_name}"
            )

            # Get basic collection info
            collection_info = await self.client.get_collection(collection_name)
            indexed_fields = await self.list_payload_indexes(collection_name)

            # Calculate index efficiency metrics
            total_points = collection_info.points_count or 0
            index_overhead = (
                len(indexed_fields) * 0.1
            )  # Estimated 10% overhead per index

            usage_stats = {
                "collection_name": collection_name,
                "collection_stats": {
                    "total_points": total_points,
                    "indexed_fields_count": len(indexed_fields),
                    "estimated_index_overhead_percent": round(index_overhead, 2),
                },
                "index_details": {},
                "performance_metrics": {
                    "expected_filter_speedup": "10-100x for indexed fields",
                    "index_selectivity": "High for keyword fields, Medium for text fields",
                    "memory_efficiency": "Optimized with proper field type mapping",
                },
                "optimization_suggestions": [],
            }

            # Categorize indexes by type for detailed analysis
            keyword_indexes = []
            text_indexes = []
            integer_indexes = []

            expected_keyword_fields = [
                "doc_type",
                "language",
                "framework",
                "version",
                "crawl_source",
                "site_name",
                "embedding_model",
                "embedding_provider",
            ]
            expected_text_fields = ["title", "content_preview"]
            expected_integer_fields = [
                "created_at",
                "last_updated",
                "word_count",
                "char_count",
            ]

            for field in indexed_fields:
                if field in expected_keyword_fields:
                    keyword_indexes.append(field)
                elif field in expected_text_fields:
                    text_indexes.append(field)
                elif field in expected_integer_fields:
                    integer_indexes.append(field)

            usage_stats["index_details"] = {
                "keyword_indexes": {
                    "count": len(keyword_indexes),
                    "fields": keyword_indexes,
                    "performance": "Excellent (exact matching, highest selectivity)",
                },
                "text_indexes": {
                    "count": len(text_indexes),
                    "fields": text_indexes,
                    "performance": "Good (full-text search, moderate selectivity)",
                },
                "integer_indexes": {
                    "count": len(integer_indexes),
                    "fields": integer_indexes,
                    "performance": "Very Good (range queries, high selectivity)",
                },
            }

            # Generate optimization suggestions
            if total_points > 100000:
                usage_stats["optimization_suggestions"].append(
                    "Large collection detected: Monitor query performance and consider composite indexes"
                )

            if len(indexed_fields) > 15:
                usage_stats["optimization_suggestions"].append(
                    "Many indexes detected: Consider removing unused indexes to reduce overhead"
                )

            if not keyword_indexes:
                usage_stats["optimization_suggestions"].append(
                    "No keyword indexes found: Add core metadata indexes for better filter performance"
                )

            if not usage_stats["optimization_suggestions"]:
                usage_stats["optimization_suggestions"].append(
                    "Index configuration is optimal for current collection size"
                )

            usage_stats["generated_at"] = int(time.time())

            return usage_stats

        except Exception as e:
            logger.error(
                f"Failed to get index usage stats for collection {collection_name}: {e}",
                exc_info=True,
            )
            msg = f"Failed to get index usage stats: {e}"
            raise QdrantServiceError(msg) from e

    def _generate_index_recommendations(
        self, missing_indexes: list[str], extra_indexes: list[str], status: str
    ) -> list[str]:
        """Generate recommendations based on index health analysis.

        Args:
            missing_indexes: Missing payload indexes
            extra_indexes: Extra payload indexes
            status: Overall health status

        Returns:
            List of recommendations

        """
        recommendations = []

        if missing_indexes:
            recommendations.append(
                f"Create missing indexes for optimal performance: {', '.join(missing_indexes)}"
            )

        if extra_indexes:
            recommendations.append(
                f"Consider removing unused indexes to reduce overhead: {', '.join(extra_indexes)}"
            )

        if status == "critical":
            recommendations.append(
                "Critical: Run migration script to create essential indexes for performance"
            )
        elif status == "warning":
            recommendations.append(
                "Warning: Some indexes are missing, performance may be suboptimal"
            )

        if not recommendations:
            recommendations.append("All indexes are healthy and optimally configured")

        return recommendations
