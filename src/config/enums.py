"""Consolidated enums for the AI Documentation Vector DB system.

This module contains all enums used across the application to ensure
consistency and avoid duplication.
"""

from enum import Enum


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level options."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""

    OPENAI = "openai"
    FASTEMBED = "fastembed"


class CrawlProvider(str, Enum):
    """Available crawling providers."""

    CRAWL4AI = "crawl4ai"
    FIRECRAWL = "firecrawl"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    BASIC = "basic"
    ENHANCED = "enhanced"
    AST = "ast"


class SearchStrategy(str, Enum):
    """Search strategy options."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class QualityTier(str, Enum):
    """Quality tiers for project configuration."""

    ECONOMY = "economy"
    BALANCED = "balanced"
    PREMIUM = "premium"


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CollectionStatus(str, Enum):
    """Vector collection status."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# Re-export commonly used enums for backward compatibility
__all__ = [
    "ChunkingStrategy",
    "CollectionStatus",
    "CrawlProvider",
    "DocumentStatus",
    "EmbeddingProvider",
    "Environment",
    "LogLevel",
    "QualityTier",
    "SearchStrategy",
]
