"""Cache metrics shim exposing the unified :class:`CacheStats` dataclass."""

from .persistent_cache import CacheStats


__all__ = ["CacheStats"]
