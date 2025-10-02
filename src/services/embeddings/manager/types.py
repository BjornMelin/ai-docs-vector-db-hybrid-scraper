"""Shared enums and dataclasses for embedding manager components."""

from __future__ import annotations

from enum import Enum


class QualityTier(Enum):
    """Defines quality tiers for embedding selection."""

    FAST = "fast"  # Uses local models for low latency
    BALANCED = "balanced"  # Balances latency and embedding quality
    BEST = "best"  # Uses high-quality models, potentially higher latency or cost
