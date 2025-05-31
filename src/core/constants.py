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

# Rate limiting defaults (calls per minute)
RATE_LIMITS = {
    "openai": {"max_calls": 500, "time_window": 60},
    "firecrawl": {"max_calls": 100, "time_window": 60},
    "crawl4ai": {"max_calls": 50, "time_window": 1},  # per second
    "qdrant": {"max_calls": 100, "time_window": 1},  # per second
}

# Cache configuration
CACHE_KEYS = {
    "embeddings": "embeddings:{model}:{hash}",
    "crawl": "crawl:{url_hash}",
    "search": "search:{query_hash}",
    "hyde": "hyde:{query_hash}",
}

CACHE_TTL_SECONDS = {
    "embeddings": 86400,  # 24 hours
    "crawl": 3600,  # 1 hour
    "search": 7200,  # 2 hours
    "hyde": 3600,  # 1 hour
}

# HNSW index configuration
HNSW_DEFAULTS = {
    "m": 16,  # Number of bi-directional links created for every new element
    "ef_construct": 200,  # Size of the dynamic candidate list
    "ef": 100,  # Size of the dynamic candidate list for search
    "max_m": 64,  # Maximum number of connections per element
    "max_m0": 32,  # Maximum number of connections for layer 0
}

# Collection-specific HNSW configurations
COLLECTION_HNSW_CONFIGS = {
    "api_reference": {"m": 20, "ef_construct": 300, "ef": 150},
    "tutorials": {"m": 16, "ef_construct": 200, "ef": 100},
    "blog_posts": {"m": 12, "ef_construct": 150, "ef": 75},
    "code_examples": {"m": 18, "ef_construct": 250, "ef": 125},
    "general": {"m": 16, "ef_construct": 200, "ef": 100},
}

# Search accuracy levels and corresponding HNSW parameters
SEARCH_ACCURACY_PARAMS = {
    "fast": {"ef": 50, "exact": False},
    "balanced": {"ef": 100, "exact": False},
    "accurate": {"ef": 200, "exact": False},
    "exact": {"exact": True},
}

# Prefetch multipliers by vector type
PREFETCH_MULTIPLIERS = {
    "dense": 2.0,
    "sparse": 5.0,
    "hyde": 3.0,
}

# Maximum prefetch limits to prevent performance degradation
MAX_PREFETCH_LIMITS = {
    "dense": 200,
    "sparse": 500,
    "hyde": 150,
}

# Chunking configuration
CHUNKING_DEFAULTS = {
    "chunk_size": 1600,
    "chunk_overlap": 320,
    "min_chunk_size": 100,
    "max_chunk_size": 3000,
    "max_function_chunk_size": 3200,
}

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
    "def", "class", "import", "return", "if", "else", "for", "while", "try", "except",
    "function", "const", "let", "var", "public", "private", "protected", "static",
    "async", "await", "yield", "lambda", "with", "as", "from", "package", "interface",
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
COMMON_VECTOR_DIMENSIONS = [
    128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096
]

# HTTP status codes
HTTP_STATUS = {
    "OK": 200,
    "CREATED": 201,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "TOO_MANY_REQUESTS": 429,
    "INTERNAL_SERVER_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503,
}

# Logging levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# Environment types
ENVIRONMENTS = ["development", "testing", "staging", "production"]

# Collection status types
COLLECTION_STATUSES = ["green", "yellow", "red"]

# Document status types
DOCUMENT_STATUSES = ["pending", "processing", "completed", "failed"]

__all__ = [
    "BUDGET_CRITICAL_THRESHOLD",
    "BUDGET_WARNING_THRESHOLD",
    "CACHE_KEYS",
    "CACHE_TTL_SECONDS",
    "CHUNKING_DEFAULTS",
    "CODE_KEYWORDS",
    "COLLECTION_HNSW_CONFIGS",
    "COLLECTION_STATUSES",
    "COMMON_VECTOR_DIMENSIONS",
    "COST_THRESHOLDS",
    "DEFAULT_CACHE_TTL",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_URLS",
    "DEFAULT_VECTOR_DIMENSIONS",
    "DOCUMENT_STATUSES",
    "EMBEDDING_BATCH_SIZE",
    "ENVIRONMENTS",
    "GC_THRESHOLD",
    "HNSW_DEFAULTS",
    "HTTP_STATUS",
    "LOG_LEVELS",
    "LONG_TEXT_THRESHOLD",
    "MAX_CONCURRENT_REQUESTS",
    "MAX_CONTENT_LENGTH",
    "MAX_DUPLICATE_RATIO",
    "MAX_MEMORY_USAGE_MB",
    "MAX_PREFETCH_LIMITS",
    "MAX_RETRIES",
    "MAX_RETRY_DELAY",
    "MAX_SEARCH_LIMIT",
    "MAX_VECTOR_DIMENSIONS",
    "MIN_CONTENT_LENGTH",
    "MIN_VECTOR_DIMENSIONS",
    "MIN_WORD_COUNT",
    "PREFETCH_MULTIPLIERS",
    "PROGRAMMING_LANGUAGES",
    "QUALITY_THRESHOLDS",
    "RATE_LIMITS",
    "SEARCH_ACCURACY_PARAMS",
    "SHORT_TEXT_THRESHOLD",
    "SPEED_THRESHOLDS",
    "SUPPORTED_EXTENSIONS",
]
