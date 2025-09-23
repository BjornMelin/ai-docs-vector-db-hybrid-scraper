"""Constants and configuration values for the AI Documentation Vector DB system.

This module centralizes all constants, default values, and magic numbers
used throughout the application.
"""

# Default timeouts and limits
DEFAULT_REQUEST_TIMEOUT = 30.0  # seconds
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
DEFAULT_CHUNK_SIZE = 1600  # characters
DEFAULT_CHUNK_OVERLAP = 320  # characters (20% of chunk size)

# Retry and circuit breaker defaults
MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 60.0  # seconds
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60.0  # seconds

# Embedding and vector search defaults
EMBEDDING_BATCH_SIZE = 32
MAX_VECTOR_DIMENSIONS = 3072
DEFAULT_VECTOR_DIMENSIONS = 1536
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100

# Performance and memory limits
MAX_CONCURRENT_REQUESTS = 10
MAX_MEMORY_USAGE_MB = 1000.0
GC_THRESHOLD = 0.8  # 80%


# Language detection patterns
PROGRAMMING_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "java",
    "cpp",
    "c",
    "csharp",
    "go",
    "rust",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "clojure",
    "haskell",
    "r",
    "matlab",
    "sql",
    "html",
    "css",
    "xml",
    "json",
    "yaml",
    "markdown",
    "dockerfile",
    "bash",
    "powershell",
]

# Code detection keywords
CODE_KEYWORDS = {
    "def",
    "class",
    "import",
    "return",
    "if",
    "else",
    "for",
    "while",
    "try",
    "except",
    "function",
    "const",
    "let",
    "var",
    "public",
    "private",
    "protected",
    "static",
    "async",
    "await",
    "yield",
    "lambda",
    "with",
    "as",
    "from",
    "package",
    "interface",
}

# API endpoints and URLs
DEFAULT_URLS = {
    "qdrant": "http://localhost:6333",
    "dragonfly": "redis://localhost:6379",
    "firecrawl": "https://api.firecrawl.dev",
}

# File extensions and MIME types
SUPPORTED_EXTENSIONS = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".rst": "text/x-rst",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
}

# Content filters
MIN_CONTENT_LENGTH = 50
MAX_CONTENT_LENGTH = 1_000_000
MIN_WORD_COUNT = 10
MAX_DUPLICATE_RATIO = 0.8

# Quality thresholds for smart model selection
QUALITY_THRESHOLDS = {
    "fast": 60.0,
    "balanced": 75.0,
    "best": 85.0,
}

# Speed thresholds (tokens/second)
SPEED_THRESHOLDS = {
    "fast": 500.0,
    "balanced": 200.0,
    "slow": 100.0,
}

# Cost thresholds (per million tokens)
COST_THRESHOLDS = {
    "cheap": 50.0,
    "moderate": 100.0,
    "expensive": 200.0,
}

# Budget management
BUDGET_WARNING_THRESHOLD = 0.8  # 80%
BUDGET_CRITICAL_THRESHOLD = 0.9  # 90%

# Text analysis thresholds
SHORT_TEXT_THRESHOLD = 100  # characters
LONG_TEXT_THRESHOLD = 2000  # characters

# Vector dimension bounds
MIN_VECTOR_DIMENSIONS = 50
MAX_VECTOR_DIMENSIONS = 10000

# Common vector dimensions for validation
COMMON_VECTOR_DIMENSIONS = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

__all__ = [
    # Budget and performance thresholds
    "BUDGET_CRITICAL_THRESHOLD",
    "BUDGET_WARNING_THRESHOLD",
    # Content validation and processing
    "CODE_KEYWORDS",
    "COMMON_VECTOR_DIMENSIONS",
    # Quality and cost thresholds
    "COST_THRESHOLDS",
    # Basic request and resource limits
    "DEFAULT_CACHE_TTL",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_URLS",
    "DEFAULT_VECTOR_DIMENSIONS",
    "EMBEDDING_BATCH_SIZE",
    "GC_THRESHOLD",
    "LONG_TEXT_THRESHOLD",
    "MAX_CONCURRENT_REQUESTS",
    "MAX_CONTENT_LENGTH",
    "MAX_DUPLICATE_RATIO",
    "MAX_MEMORY_USAGE_MB",
    "MAX_RETRIES",
    "MAX_RETRY_DELAY",
    "MAX_SEARCH_LIMIT",
    "MAX_VECTOR_DIMENSIONS",
    "MIN_CONTENT_LENGTH",
    "MIN_VECTOR_DIMENSIONS",
    "MIN_WORD_COUNT",
    "PROGRAMMING_LANGUAGES",
    "QUALITY_THRESHOLDS",
    "SHORT_TEXT_THRESHOLD",
    "SPEED_THRESHOLDS",
    "SUPPORTED_EXTENSIONS",
]
