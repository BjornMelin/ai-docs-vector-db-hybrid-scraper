"""Compatibility layer for legacy retrieval model imports.

This module now re-exports the canonical search models from
``src.models.search``. Runtime code should migrate to the new module to
avoid relying on the deprecated contracts namespace.
"""

from __future__ import annotations

from src.models.search import SearchRecord, SearchResponse


__all__ = ["SearchRecord", "SearchResponse"]
