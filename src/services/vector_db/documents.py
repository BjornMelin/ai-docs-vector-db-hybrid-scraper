"""Focused QdrantDocuments module for document/point operations.

This module provides a clean, focused implementation of document operations
extracted from QdrantService, focusing specifically on point CRUD operations.
"""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from ...config import UnifiedConfig
from ..errors import QdrantServiceError

logger = logging.getLogger(__name__)


class QdrantDocuments:
    """Focused document/point operations for Qdrant with batching optimization."""

    def __init__(self, client: AsyncQdrantClient, config: UnifiedConfig):
        """Initialize documents service.

        Args:
            client: Initialized Qdrant client
            config: Unified configuration
        """
        self.client = client
        self.config = config

    async def upsert_points(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> bool:
        """Upsert points with automatic batching.

        Args:
            collection_name: Target collection
            points: List of points with id, vector, and optional payload
            batch_size: Points per batch for memory efficiency

        Returns:
            True if all points upserted successfully

        Raises:
            QdrantServiceError: If upsert fails
        """
        try:
            # Process in batches
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]

                # Convert to PointStruct
                point_structs = []
                for point in batch:
                    vectors = point.get("vector", {})
                    if isinstance(vectors, list):
                        vectors = {"dense": vectors}

                    point_struct = models.PointStruct(
                        id=point["id"],
                        vector=vectors,
                        payload=point.get("payload", {}),
                    )
                    point_structs.append(point_struct)

                # Upsert batch
                await self.client.upsert(
                    collection_name=collection_name,
                    points=point_structs,
                    wait=True,
                )

                logger.info(
                    f"Upserted batch {i // batch_size + 1} "
                    f"({len(point_structs)} points)"
                )

            return True

        except Exception as e:
            logger.error(
                f"Failed to upsert {len(points)} points to {collection_name}: {e}",
                exc_info=True,
            )

            error_msg = str(e).lower()
            if "collection not found" in error_msg:
                raise QdrantServiceError(
                    f"Collection '{collection_name}' not found. Create it before upserting."
                ) from e
            elif "wrong vector size" in error_msg:
                raise QdrantServiceError(
                    "Vector dimension mismatch. Check that vectors match collection configuration."
                ) from e
            elif "payload too large" in error_msg:
                raise QdrantServiceError(
                    f"Payload too large. Try reducing batch size (current: {batch_size})."
                ) from e
            else:
                raise QdrantServiceError(f"Failed to upsert points: {e}") from e

    async def get_points(
        self,
        collection_name: str,
        point_ids: list[str | int],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve specific points by their IDs.

        Args:
            collection_name: Collection to search
            point_ids: List of point IDs to retrieve
            with_payload: Include payload in results
            with_vectors: Include vectors in results

        Returns:
            List of retrieved points

        Raises:
            QdrantServiceError: If retrieval fails
        """
        try:
            # Retrieve points
            results = await self.client.retrieve(
                collection_name=collection_name,
                ids=point_ids,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            # Format results
            formatted_results = []
            for point in results:
                result = {
                    "id": str(point.id),
                }

                if with_payload and point.payload:
                    result["payload"] = point.payload

                if with_vectors and point.vector:
                    result["vector"] = point.vector

                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(
                f"Failed to retrieve points from {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to retrieve points: {e}") from e

    async def delete_points(
        self,
        collection_name: str,
        point_ids: list[str | int] | None = None,
        filter_condition: dict[str, Any] | None = None,
    ) -> bool:
        """Delete points by IDs or filter condition.

        Args:
            collection_name: Collection to delete from
            point_ids: Specific point IDs to delete (if provided)
            filter_condition: Filter condition for bulk deletion (if provided)

        Returns:
            True if deletion successful

        Raises:
            QdrantServiceError: If deletion fails
        """
        if not point_ids and not filter_condition:
            raise ValueError("Either point_ids or filter_condition must be provided")

        try:
            if point_ids:
                # Delete by specific IDs
                await self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                    wait=True,
                )
                logger.info(f"Deleted {len(point_ids)} points by ID")
            else:
                # Delete by filter
                filter_obj = self._build_filter(filter_condition)
                await self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=filter_obj),
                    wait=True,
                )
                logger.info("Deleted points by filter condition")

            return True

        except Exception as e:
            logger.error(
                f"Failed to delete points from {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to delete points: {e}") from e

    async def update_point_payload(
        self,
        collection_name: str,
        point_id: str | int,
        payload: dict[str, Any],
        replace: bool = False,
    ) -> bool:
        """Update payload for a specific point.

        Args:
            collection_name: Collection containing the point
            point_id: ID of the point to update
            payload: New payload data
            replace: If True, replace entire payload; if False, merge with existing

        Returns:
            True if update successful

        Raises:
            QdrantServiceError: If update fails
        """
        try:
            if replace:
                # Replace entire payload
                await self.client.overwrite_payload(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=[point_id]),
                    payload=payload,
                    wait=True,
                )
                logger.info(f"Replaced payload for point {point_id}")
            else:
                # Merge with existing payload
                await self.client.set_payload(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=[point_id]),
                    payload=payload,
                    wait=True,
                )
                logger.info(f"Updated payload for point {point_id}")

            return True

        except Exception as e:
            logger.error(
                f"Failed to update payload for point {point_id}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to update point payload: {e}") from e

    async def count_points(
        self,
        collection_name: str,
        filter_condition: dict[str, Any] | None = None,
        exact: bool = True,
    ) -> int:
        """Count points in collection with optional filtering.

        Args:
            collection_name: Collection name
            filter_condition: Optional filter to apply
            exact: Use exact count

        Returns:
            Point count

        Raises:
            QdrantServiceError: If counting fails
        """
        try:
            filter_obj = (
                self._build_filter(filter_condition) if filter_condition else None
            )

            result = await self.client.count(
                collection_name=collection_name,
                count_filter=filter_obj,
                exact=exact,
            )
            return result.count

        except Exception as e:
            logger.error(
                f"Failed to count points in {collection_name}: {e}", exc_info=True
            )
            raise QdrantServiceError(f"Failed to count points: {e}") from e

    async def scroll_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | int | None = None,
        filter_condition: dict[str, Any] | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> dict[str, Any]:
        """Scroll through points in a collection with pagination.

        Args:
            collection_name: Collection to scroll
            limit: Maximum number of points to return
            offset: Offset for pagination (point ID or numeric offset)
            filter_condition: Optional filter to apply
            with_payload: Include payload in results
            with_vectors: Include vectors in results

        Returns:
            Dictionary with points and next offset

        Raises:
            QdrantServiceError: If scrolling fails
        """
        try:
            filter_obj = (
                self._build_filter(filter_condition) if filter_condition else None
            )

            result = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_obj,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            # Format results
            formatted_points = []
            for point in result[0]:  # result is tuple (points, next_offset)
                formatted_point = {"id": str(point.id)}

                if with_payload and point.payload:
                    formatted_point["payload"] = point.payload

                if with_vectors and point.vector:
                    formatted_point["vector"] = point.vector

                formatted_points.append(formatted_point)

            return {
                "points": formatted_points,
                "next_offset": result[1],  # Next offset for pagination
            }

        except Exception as e:
            logger.error(
                f"Failed to scroll points in {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to scroll points: {e}") from e

    async def clear_collection(self, collection_name: str) -> bool:
        """Clear all points from a collection without deleting the collection.

        Args:
            collection_name: Collection to clear

        Returns:
            True if successful

        Raises:
            QdrantServiceError: If clearing fails
        """
        try:
            # Delete all points using an empty filter (matches everything)
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()  # Empty filter matches all points
                ),
                wait=True,
            )

            logger.info(f"Cleared all points from collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to clear collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to clear collection: {e}") from e

    def _build_filter(self, filters: dict[str, Any] | None) -> models.Filter | None:
        """Build optimized Qdrant filter from filter dictionary using indexed fields.

        Args:
            filters: Filter conditions dictionary

        Returns:
            Qdrant Filter object or None
        """
        if not filters:
            return None

        conditions = []

        # Keyword filters for exact matching
        keyword_fields = [
            "doc_type",
            "language",
            "framework",
            "version",
            "crawl_source",
            "site_name",
            "embedding_model",
            "embedding_provider",
            "url",
        ]
        for field in keyword_fields:
            if field in filters:
                if not isinstance(filters[field], str | int | float | bool):
                    raise ValueError(f"Filter value for {field} must be a simple type")
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=filters[field])
                    )
                )

        # Text filters for partial matching
        for field in ["title", "content_preview"]:
            if field in filters:
                if not isinstance(filters[field], str):
                    raise ValueError(f"Text filter value for {field} must be a string")
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchText(text=filters[field])
                    )
                )

        # Range filters for timestamps and metrics
        range_mappings = [
            ("created_after", "created_at", "gte"),
            ("created_before", "created_at", "lte"),
            ("updated_after", "last_updated", "gte"),
            ("updated_before", "last_updated", "lte"),
            ("scraped_after", "scraped_at", "gte"),
            ("scraped_before", "scraped_at", "lte"),
            ("min_word_count", "word_count", "gte"),
            ("max_word_count", "word_count", "lte"),
            ("min_char_count", "char_count", "gte"),
            ("max_char_count", "char_count", "lte"),
            ("min_quality_score", "quality_score", "gte"),
            ("max_quality_score", "quality_score", "lte"),
            ("min_score", "score", "gte"),
            ("min_total_chunks", "total_chunks", "gte"),
            ("max_total_chunks", "total_chunks", "lte"),
            ("min_links_count", "links_count", "gte"),
            ("max_links_count", "links_count", "lte"),
        ]

        for filter_key, field_key, operator in range_mappings:
            if filter_key in filters:
                range_params = {operator: filters[filter_key]}
                conditions.append(
                    models.FieldCondition(
                        key=field_key, range=models.Range(**range_params)
                    )
                )

        # Exact match filters for structural fields
        for field in ["chunk_index", "depth"]:
            if field in filters:
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=filters[field])
                    )
                )

        return models.Filter(must=conditions) if conditions else None
