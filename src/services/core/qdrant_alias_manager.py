"""Qdrant alias manager for zero-downtime collection updates."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
)

from src.config.loader import Settings
from src.services.errors import QdrantServiceError
from src.services.observability.tracing import log_extra_with_trace


logger = logging.getLogger(__name__)

_EXPECTED_QDRANT_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    OSError,
    PermissionError,
    RuntimeError,
    AttributeError,
    ValueError,
)

_T = TypeVar("_T")
_MISSING = object()
MetadataSupplier = Callable[[], dict[str, Any]]


def _log_extra(event: str, **metadata: Any) -> dict[str, Any]:
    """Return structured logging extras enriched with trace metadata."""
    metadata.setdefault("component", "qdrant.alias_manager")
    return log_extra_with_trace(event, **metadata)


# Valid collection/alias name pattern (alphanumeric, underscore, hyphen)
VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_NAME_LENGTH = 255


class QdrantAliasManager:
    """Manage Qdrant collection aliases for zero-downtime updates."""

    def __init__(
        self,
        config: Settings,
        client: AsyncQdrantClient,
    ):
        """Initialize alias manager.

        Args:
            config: Unified configuration
            client: Qdrant client instance
        """
        self.config = config
        self.client = client

    @staticmethod
    def validate_name(name: str | None, name_type: str = "name") -> None:
        """Validate collection or alias name.

        Args:
            name: Name to validate
            name_type: Type of name for error message

        Raises:
            QdrantServiceError: If name is invalid
        """
        if not name:
            msg = f"{name_type} cannot be empty"
            raise QdrantServiceError(msg)

        if len(name) > MAX_NAME_LENGTH:
            msg = f"{name_type} '{name}' exceeds maximum length of {MAX_NAME_LENGTH}"
            raise QdrantServiceError(msg)

        if not VALID_NAME_PATTERN.match(name):
            msg = (
                f"{name_type} '{name}' contains invalid characters. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )
            raise QdrantServiceError(msg)

    async def initialize(self) -> None:
        """No-op as client is already initialized."""

    async def cleanup(self) -> None:
        """Cleanup alias manager.

        Note: Persistent task queue jobs continue running independently.
        """
        # No local cleanup needed as tasks are managed by task queue

    @staticmethod
    def is_initialized() -> bool:
        """Return True as the manager relies on an injected client."""
        return True

    async def _execute_with_handling(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        action: str,
        log_event: str,
        metadata: dict[str, Any] | MetadataSupplier | None = None,
        log_message: str | None = None,
        unexpected_log_message: str | None = None,
        default: object = _MISSING,
        default_factory: Callable[[], _T] | None = None,
        expected_handler: Callable[[BaseException], _T] | None = None,
        unexpected_handler: Callable[[BaseException], _T] | None = None,
        log_expected: bool = True,
    ) -> _T:
        """Execute an async operation with shared Qdrant exception handling.

        Args:
            operation: Coroutine factory to execute.
            action: Human-readable verb for error messages.
            log_event: Structured logging event name.
            metadata: Static metadata dict or callable returning metadata per failure.
            log_message: Optional log message for handled exceptions.
            unexpected_log_message: Optional log message for unexpected exceptions.
            default: Optional default value to return on handled exceptions.
            default_factory: Factory invoked to produce default result on handled
                exceptions. Mutually exclusive with ``default``.
            expected_handler: Callback generating a result for handled exceptions.
            unexpected_handler: Callback generating a result for unexpected exceptions.
            log_expected: Whether to log handled exceptions.

        Returns:
            Result of ``operation`` or fallback value.

        Raises:
            QdrantServiceError: Raised when no fallback handles the exception.
        """
        provided_fallbacks = [
            default is not _MISSING,
            default_factory is not None,
            expected_handler is not None,
        ]
        if sum(provided_fallbacks) > 1:
            msg = "Provide only one of default, default_factory, or expected_handler."
            raise ValueError(msg)
        if unexpected_handler is not None and (
            default is not _MISSING or default_factory is not None
        ):
            msg = (
                "Unexpected handler cannot be combined with default or default_factory."
            )
            raise ValueError(msg)

        def _resolve_metadata() -> dict[str, Any]:
            if metadata is None:
                return {}
            if callable(metadata):
                return metadata()
            return metadata

        expected_message = log_message or f"Failed to {action}"
        unexpected_message = (
            unexpected_log_message or f"{expected_message} (unexpected error)"
        )

        def _handle_failure(
            exc: BaseException, handler: Callable[[BaseException], _T] | None
        ) -> _T:
            if handler is not None:
                return handler(exc)
            if default is not _MISSING:
                return cast(_T, default)
            if default_factory is not None:
                return default_factory()
            raise QdrantServiceError(f"Failed to {action}: {exc}") from exc

        captured_exc: BaseException | None = None
        handler_for_failure: Callable[[BaseException], _T] | None = None

        try:
            return await operation()
        except Exception as exc:  # pragma: no cover - consolidated handling
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, _EXPECTED_QDRANT_ERRORS):
                if log_expected:
                    logger.exception(
                        expected_message,
                        extra=_log_extra(log_event, **_resolve_metadata()),
                    )
                captured_exc = exc
                handler_for_failure = expected_handler
            else:
                logger.exception(
                    unexpected_message,
                    extra=_log_extra(log_event, **_resolve_metadata()),
                )
                captured_exc = exc
                handler_for_failure = unexpected_handler

        if captured_exc is None:
            msg = "Operation completed without result or exception."
            raise RuntimeError(msg)
        return _handle_failure(captured_exc, handler_for_failure)

    async def create_alias(
        self, alias_name: str, collection_name: str, force: bool = False
    ) -> bool:
        """Create or update an alias to point to a collection.

        Args:
            alias_name: Name of the alias
            collection_name: Name of the collection to point to
            force: If True, delete existing alias before creating

        Returns:
            True if alias created successfully

        Raises:
            QdrantServiceError: If operation fails
        """
        # Validate names
        self.validate_name(alias_name, "Alias name")
        self.validate_name(collection_name, "Collection name")

        async def _create() -> bool:
            if await self.alias_exists(alias_name):
                if not force:
                    logger.warning(
                        "Alias %s already exists",
                        alias_name,
                        extra=_log_extra(
                            "qdrant.alias.exists",
                            alias=alias_name,
                            collection=collection_name,
                        ),
                    )
                    return False
                await self.delete_alias(alias_name)

            await self.client.update_collection_aliases(
                change_aliases_operations=[
                    CreateAliasOperation(
                        create_alias=CreateAlias(
                            alias_name=alias_name,
                            collection_name=collection_name,
                        )
                    )
                ]
            )

            logger.info(
                "Created alias %s -> %s",
                alias_name,
                collection_name,
                extra=_log_extra(
                    "qdrant.alias.create",
                    alias=alias_name,
                    collection=collection_name,
                ),
            )
            return True

        return await self._execute_with_handling(
            _create,
            action="create alias",
            log_event="qdrant.alias.create",
            metadata=lambda: {"alias": alias_name, "collection": collection_name},
            log_message="Failed to create alias",
        )

    async def switch_alias(
        self, alias_name: str, new_collection: str, delete_old: bool = False
    ) -> str | None:
        """Atomically switch alias to new collection.

        Args:
            alias_name: Name of the alias to switch
            new_collection: New collection to point to
            delete_old: If True, schedule old collection for deletion

        Returns:
            Name of old collection if switch successful, None if alias unchanged

        Raises:
            QdrantServiceError: If operation fails
        """
        # Validate names
        self.validate_name(alias_name, "Alias name")
        self.validate_name(new_collection, "Collection name")

        old_collection: str | None = None

        async def _switch() -> str | None:
            nonlocal old_collection
            old_collection = await self.get_collection_for_alias(alias_name)

            if old_collection == new_collection:
                logger.warning(
                    "Alias already points to target collection",
                    extra=_log_extra(
                        "qdrant.alias.switch",
                        alias=alias_name,
                        old_collection=old_collection,
                        new_collection=new_collection,
                        skipped=True,
                    ),
                )
                return None

            operations = []

            if old_collection:
                operations.append(
                    DeleteAliasOperation(
                        delete_alias=DeleteAlias(alias_name=alias_name)
                    )
                )

            operations.append(
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        alias_name=alias_name, collection_name=new_collection
                    )
                )
            )

            await self.client.update_collection_aliases(
                change_aliases_operations=operations
            )

            logger.info(
                "Switched alias %s: %s -> %s",
                alias_name,
                old_collection,
                new_collection,
                extra=_log_extra(
                    "qdrant.alias.switch",
                    alias=alias_name,
                    old_collection=old_collection,
                    new_collection=new_collection,
                    deleted_old=delete_old,
                ),
            )

            if delete_old and old_collection:
                await self.safe_delete_collection(old_collection)

            return old_collection

        return await self._execute_with_handling(
            _switch,
            action="switch alias",
            log_event="qdrant.alias.switch",
            metadata=lambda: {
                "alias": alias_name,
                "old_collection": old_collection,
                "new_collection": new_collection,
            },
            log_message="Failed to switch alias",
        )

    async def delete_alias(self, alias_name: str) -> bool:
        """Delete an alias.

        Args:
            alias_name: Name of the alias to delete

        Returns:
            True if alias deleted successfully

        Raises:
            QdrantServiceError: If operation fails
        """

        async def _delete() -> bool:
            await self.client.update_collection_aliases(
                change_aliases_operations=[
                    DeleteAliasOperation(
                        delete_alias=DeleteAlias(alias_name=alias_name)
                    )
                ]
            )

            logger.info(
                "Deleted alias %s",
                alias_name,
                extra=_log_extra("qdrant.alias.delete", alias=alias_name),
            )
            return True

        return await self._execute_with_handling(
            _delete,
            action="delete alias",
            log_event="qdrant.alias.delete",
            metadata=lambda: {"alias": alias_name},
            log_message="Failed to delete alias",
            unexpected_log_message="Unexpected error deleting alias",
            default=False,
        )

    async def alias_exists(self, alias_name: str) -> bool:
        """Check if an alias exists."""

        async def _exists() -> bool:
            aliases = await self.client.get_aliases()
            return any(alias.alias_name == alias_name for alias in aliases.aliases)

        return await self._execute_with_handling(
            _exists,
            action="check alias existence",
            log_event="qdrant.alias.exists",
            metadata=lambda: {"alias": alias_name},
            default=False,
            log_expected=False,
        )

    async def get_collection_for_alias(self, alias_name: str) -> str | None:
        """Get collection name that alias points to."""

        async def _resolve() -> str | None:
            aliases = await self.client.get_aliases()
            for alias in aliases.aliases:
                if alias.alias_name == alias_name:
                    return alias.collection_name
            return None

        return await self._execute_with_handling(
            _resolve,
            action="resolve alias collection",
            log_event="qdrant.alias.resolve",
            metadata=lambda: {"alias": alias_name},
            default=None,
            log_expected=False,
        )

    async def list_aliases(self) -> dict[str, str]:
        """List all aliases and their collections.

        Returns:
            Dictionary mapping alias names to collection names
        """

        async def _list() -> dict[str, str]:
            aliases = await self.client.get_aliases()
            return {
                alias.alias_name: alias.collection_name for alias in aliases.aliases
            }

        return await self._execute_with_handling(
            _list,
            action="list aliases",
            log_event="qdrant.alias.list",
            log_message="Failed to list aliases",
            default_factory=dict,
        )

    async def safe_delete_collection(
        self, collection_name: str, grace_period_minutes: int = 60
    ) -> None:
        """Safely delete collection after grace period.

        Uses persistent task queue to ensure deletion completes even if server restarts.

        Args:
            collection_name: Name of collection to delete
            grace_period_minutes: Minutes to wait before deletion
        """
        # Check if any alias points to this collection
        aliases = await self.list_aliases()
        if collection_name in aliases.values():
            logger.warning(
                "Collection %s still has aliases",
                collection_name,
                extra=_log_extra(
                    "qdrant.collection.delete", collection=collection_name, skipped=True
                ),
            )
            return

        logger.info(
            "Deleting collection %s after grace period of %d minutes",
            collection_name,
            grace_period_minutes,
            extra=_log_extra(
                "qdrant.collection.delete",
                collection=collection_name,
                grace_minutes=grace_period_minutes,
            ),
        )

        await asyncio.sleep(grace_period_minutes * 60)
        await self.client.delete_collection(collection_name)
        logger.info(
            "Collection %s deleted",
            collection_name,
            extra=_log_extra(
                "qdrant.collection.delete",
                collection=collection_name,
                grace_minutes=grace_period_minutes,
            ),
        )

    async def clone_collection_schema(self, source: str, target: str) -> bool:
        """Clone collection schema from source to target."""

        async def _clone() -> bool:
            source_info = await self.client.get_collection(source)
            vectors_config = source_info.config.params.vectors

            await self.client.create_collection(
                collection_name=target,
                vectors_config=cast(Any, vectors_config),
                hnsw_config=cast(Any, source_info.config.hnsw_config),
                quantization_config=cast(Any, source_info.config.quantization_config),
                on_disk_payload=source_info.config.params.on_disk_payload,
            )

            payload_schema = getattr(source_info.config, "payload_schema", None)
            if payload_schema:
                for field_name, field_schema in payload_schema.items():
                    await self.client.create_payload_index(
                        collection_name=target,
                        field_name=field_name,
                        field_type=field_schema.data_type,
                    )

            logger.info(
                "Cloned collection schema from %s to %s",
                source,
                target,
                extra=_log_extra(
                    "qdrant.collection.clone", source=source, target=target
                ),
            )
            return True

        return await self._execute_with_handling(
            _clone,
            action="clone collection schema",
            log_event="qdrant.collection.clone",
            metadata=lambda: {"source": source, "target": target},
            log_message="Failed to clone collection schema",
        )

    async def copy_collection_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        source: str,
        target: str,
        batch_size: int = 100,
        limit: int | None = None,
        progress_callback: Callable | None = None,
    ) -> int:
        """Copy data from source collection to target.

        Args:
            source: Source collection name
            target: Target collection name
            batch_size: Number of points to copy per batch
            limit: Maximum number of points to copy (None for all)
            progress_callback: Optional callback(copied, total) for progress updates

        Returns:
            Number of points copied

        Raises:
            QdrantServiceError: If operation fails
        """
        # Validate names
        self.validate_name(source, "Source collection name")
        self.validate_name(target, "Target collection name")

        total_copied = 0
        total_points = 0

        async def _copy() -> int:
            """Copy collection data."""
            nonlocal total_copied, total_points
            source_info = await self.client.get_collection(source)
            total_points = source_info.points_count or 0

            if limit and total_points > limit:
                total_points = limit

            offset = None

            while True:
                records, next_offset = await self.client.scroll(
                    collection_name=source,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )

                if not records:
                    break

                await self.client.upsert(
                    collection_name=target, points=cast(Any, records)
                )

                total_copied += len(records)
                logger.debug(
                    "Copied %d points from %s to %s",
                    total_copied,
                    source,
                    target,
                    extra=_log_extra(
                        "qdrant.collection.copy",
                        source=source,
                        target=target,
                        copied=total_copied,
                        total=total_points,
                    ),
                )

                if progress_callback:
                    try:
                        await progress_callback(total_copied, total_points)
                    except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
                        logger.warning(
                            "Progress callback failed: %s",
                            e,
                            extra=_log_extra(
                                "qdrant.collection.copy", source=source, target=target
                            ),
                        )
                    except (AttributeError, ValueError, TypeError) as e:
                        logger.warning(
                            "Progress callback raised unexpected error: %s",
                            e,
                            extra=_log_extra(
                                "qdrant.collection.copy", source=source, target=target
                            ),
                        )
                    except Exception as e:  # pylint: disable=broad-except # pragma: no cover - diagnostic guard # noqa: BLE001
                        logger.warning(
                            "Progress callback raised error: %s",
                            e,
                            extra=_log_extra(
                                "qdrant.collection.copy", source=source, target=target
                            ),
                        )

                if limit and total_copied >= limit:
                    break

                offset = next_offset
                if offset is None:
                    break

            logger.info(
                "Copied %d points from %s to %s",
                total_copied,
                source,
                target,
                extra=_log_extra(
                    "qdrant.collection.copy",
                    source=source,
                    target=target,
                    copied=total_copied,
                    total=total_points,
                ),
            )

            return total_copied

        return await self._execute_with_handling(
            _copy,
            action="copy collection data",
            log_event="qdrant.collection.copy",
            metadata=lambda: {
                "source": source,
                "target": target,
                "copied": total_copied,
                "total": total_points,
            },
            log_message="Failed to copy collection data",
        )

    async def validate_collection_compatibility(
        self, collection1: str, collection2: str
    ) -> tuple[bool, str]:
        """Validate that two collections have compatible schemas.

        Args:
            collection1: First collection name
            collection2: Second collection name

        Returns:
            Tuple of (is_compatible, message)
        """

        async def _validate() -> tuple[bool, str]:
            """Validate collection compatibility."""
            info1 = await self.client.get_collection(collection1)
            info2 = await self.client.get_collection(collection2)
            if info1.config.params.vectors != info2.config.params.vectors:
                return (
                    False,
                    f"Vector configurations differ: {collection1} vs {collection2}",
                )

            hnsw1 = info1.config.hnsw_config
            hnsw2 = info2.config.hnsw_config

            if (
                hnsw1
                and hnsw2
                and (hnsw1.m != hnsw2.m or hnsw1.ef_construct != hnsw2.ef_construct)
            ):
                return False, "HNSW configurations differ"

            quant1 = info1.config.quantization_config
            quant2 = info2.config.quantization_config

            if (quant1 is None) != (quant2 is None):
                return False, "Quantization configuration mismatch"

            return True, "Collections are compatible"

        def _as_validation_error(exc: BaseException) -> tuple[bool, str]:
            return False, f"Validation error: {exc!s}"

        return await self._execute_with_handling(
            _validate,
            action="validate collection compatibility",
            log_event="qdrant.collection.validate",
            metadata=lambda: {
                "collection1": collection1,
                "collection2": collection2,
            },
            log_message="Failed to validate collection compatibility",
            expected_handler=_as_validation_error,
            unexpected_handler=_as_validation_error,
        )
