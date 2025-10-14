"""Cache warmup utilities leveraging Dragonfly-backed caches."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from src.services.cache.manager import CacheManager


if TYPE_CHECKING:
    from src.services.embeddings.manager import EmbeddingManager


async def warm_caches(
    cache_manager: CacheManager,
    *,
    embedding_manager: EmbeddingManager | None = None,
    embedding_queries: Sequence[str] | None = None,
    search_queries: Sequence[str] | None = None,
    search_collection: str = "default",
    search_executor: Callable[[str, str], Awaitable[list[dict[str, Any]]]]
    | None = None,
) -> dict[str, Any]:
    """Warm configured caches using the provided application services.

    Args:
        cache_manager: Cache manager coordinating Dragonfly access.
        embedding_manager: Optional embedding manager used for generation.
        embedding_queries: Text queries to prime the embedding cache.
        search_queries: Popular search queries to prepopulate results.
        search_collection: Target collection name for search warming.
        search_executor: Callable executing a search against the collection.

    Returns:
        Dictionary summarising the warmup results per cache type.
    """
    summary: dict[str, Any] = {
        "embeddings": {
            "requested": len(embedding_queries or []),
            "already_cached": 0,
            "generated": 0,
            "skipped": 0,
        },
        "search": {
            "requested": len(search_queries or []),
            "warmed": 0,
            "skipped": 0,
        },
    }

    if embedding_manager is not None and embedding_queries:
        await embedding_manager.initialize()
        embedding_cache = cache_manager.embedding_cache
        if embedding_cache is None:
            summary["embeddings"]["skipped"] = len(embedding_queries)
        else:
            config = embedding_manager.config.embedding
            missing = await embedding_cache.warm_cache(
                list(embedding_queries),
                model=config.dense_model,
                provider=config.provider,
            )
            summary["embeddings"]["already_cached"] = len(embedding_queries) - len(
                missing
            )
            if missing:
                result = await embedding_manager.generate_embeddings(list(missing))
                embeddings = cast(
                    Sequence[Sequence[float]], result.get("embeddings", [])
                )
                generated = len(embeddings)
                summary["embeddings"]["generated"] = generated
                if generated < len(missing):
                    summary["embeddings"]["skipped"] = len(missing) - generated
    elif embedding_queries:
        summary["embeddings"]["skipped"] = len(embedding_queries)

    search_cache = cache_manager.search_cache
    if search_cache is not None and search_queries and search_executor is not None:
        warmed = await search_cache.warm_popular_searches(
            list(search_queries),
            collection_name=search_collection,
            search_func=search_executor,
        )
        summary["search"]["warmed"] = warmed
    elif search_queries:
        summary["search"]["skipped"] = len(search_queries)

    return summary
