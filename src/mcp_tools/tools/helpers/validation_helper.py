"""Validation helpers for query processing MCP tools."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from src.security import MLSecurityValidator
from src.services.query_processing.models import MatryoshkaDimension, SearchStrategy


class QueryValidationHelper:
    """Helper for validating query processing requests."""

    def __init__(self):
        """Initialize validation helper."""
        self.security_validator = MLSecurityValidator.from_unified_config()

    def validate_query_request(self, request) -> tuple[str, str]:
        """Validate collection and query from request."""
        validated_collection = self.security_validator.validate_collection_name(
            request.collection
        )
        validated_query = self.security_validator.validate_query_string(request.query)
        return validated_collection, validated_query

    async def validate_force_options(
        self, request, ctx: "Context"
    ) -> tuple[SearchStrategy | None, MatryoshkaDimension | None]:
        """Validate force strategy and dimension options."""
        force_strategy = None
        if request.force_strategy:
            try:
                force_strategy = SearchStrategy(request.force_strategy.lower())
            except ValueError:
                await ctx.warning(
                    f"Invalid force_strategy '{request.force_strategy}', ignoring"
                )

        force_dimension = None
        if request.force_dimension:
            try:
                # Map dimension values to MatryoshkaDimension
                dimension_map = {
                    512: MatryoshkaDimension.SMALL,
                    768: MatryoshkaDimension.MEDIUM,
                    1536: MatryoshkaDimension.LARGE,
                }
                force_dimension = dimension_map.get(request.force_dimension)
                if not force_dimension:
                    await ctx.warning(
                        f"Invalid force_dimension '{request.force_dimension}', ignoring"
                    )
            except (ConnectionError, OSError, PermissionError):
                await ctx.warning(
                    f"Invalid force_dimension '{request.force_dimension}', ignoring"
                )

        return force_strategy, force_dimension
