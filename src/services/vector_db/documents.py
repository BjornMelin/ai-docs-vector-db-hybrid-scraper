"""Focused QdrantDocuments module for document/point operations.

This module provides a clean, focused implementation of document operations
extracted from QdrantService, focusing specifically on point CRUD operations.
"""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.config import Config
from src.services.errors import QdrantServiceError

from .utils import build_filter


logger = logging.getLogger(__name__)


class QdrantDocuments:
    """Focused document/point operations for Qdrant with batching optimization."""

    def __init__(self, client: AsyncQdrantClient, config: Config):
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
                batch_number = i // batch_size + 1

                await self._upsert_single_batch(
                    collection_name, batch, batch_number, batch_size
                )

        except Exception as e:
            logger.exception("Operation failed")
            self._handle_upsert_error(e, collection_name, batch_size)
        else:
            return True

    async def _upsert_single_batch(
        self,
        collection_name: str,
        batch: list[dict[str, Any]],
        batch_number: int,
        batch_size: int,  # noqa: ARG002
    ) -> None:
        """Upsert a single batch of points."""
        try:
            point_structs = self._convert_to_point_structs(batch)
        except Exception as e:
            msg = f"Failed to convert batch {batch_number} to point structs: {e}"
            raise QdrantServiceError(msg) from e

        await self.client.upsert(
            collection_name=collection_name,
            points=point_structs,
            wait=True,
        )

        logger.info(
            "Upserted batch %d (%d points)",
            batch_number,
            len(point_structs),
        )

    def _convert_to_point_structs(
        self, batch: list[dict[str, Any]]
    ) -> list[models.PointStruct]:
        """Convert batch points to PointStruct objects."""
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

        return point_structs

    def _handle_upsert_error(
        self, e: Exception, collection_name: str, batch_size: int
    ) -> None:
        """Handle and re-raise upsert errors with appropriate messages."""
        error_msg = str(e).lower()
        if "collection not found" in error_msg:
            msg = (
                f"Collection '{collection_name}' not found. Create it before upserting."
            )
            raise QdrantServiceError(msg) from e
        if "wrong vector size" in error_msg:
            msg = (
                "Vector dimension mismatch. Check that vectors match "
                "collection configuration."
            )
            raise QdrantServiceError(msg) from e
        if "payload too large" in error_msg:
            msg = f"Payload too large. Try reducing batch size (current: {batch_size})."
            raise QdrantServiceError(msg) from e
        msg = f"Failed to upsert points: {e}"
        raise QdrantServiceError(msg) from e

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

        except Exception as e:
            logger.exception("Failed to retrieve points from %s", collection_name)
            msg = f"Failed to retrieve points: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return formatted_results

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
            msg = "Either point_ids or filter_condition must be provided"
            raise ValueError(msg)

        try:
            if point_ids:
                # Delete by specific IDs
                await self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                    wait=True,
                )
                logger.info("Deleted %d points by ID", len(point_ids))
            else:
                # Delete by filter
                filter_obj = build_filter(filter_condition)
                await self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=filter_obj),
                    wait=True,
                )
                logger.info("Deleted points by filter condition")

        except Exception as e:
            logger.exception("Failed to delete points from %s", collection_name)
            msg = f"Failed to delete points: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return True

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
                logger.info("Replaced payload for point %s", point_id)
            else:
                # Merge with existing payload
                await self.client.set_payload(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=[point_id]),
                    payload=payload,
                    wait=True,
                )
                logger.info("Updated payload for point %s", point_id)

        except Exception as e:
            logger.exception("Operation failed")
            msg = f"Failed to update point payload: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return True

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
            filter_obj = build_filter(filter_condition) if filter_condition else None

            result = await self.client.count(
                collection_name=collection_name,
                count_filter=filter_obj,
                exact=exact,
            )

        except Exception as e:
            logger.exception(
                "Failed to count points in collection: %s", collection_name
            )
            msg = f"Failed to count points: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return result.count

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
            filter_obj = build_filter(filter_condition) if filter_condition else None

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
            logger.exception("Operation failed")
            msg = f"Failed to scroll points: {e}"
            raise QdrantServiceError(msg) from e

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

            logger.info("Cleared all points from collection: %s", collection_name)

        except Exception as e:
            logger.exception("Operation failed")
            msg = f"Failed to clear collection: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return True
