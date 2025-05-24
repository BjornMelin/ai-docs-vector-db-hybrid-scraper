"""Service-specific error classes."""


class APIError(Exception):
    """Base API error."""

    def __init__(self, message: str, status_code: int | None = None):
        """Initialize API error.

        Args:
            message: Error message
            status_code: Optional HTTP status code
        """
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class QdrantServiceError(APIError):
    """Qdrant-specific errors."""

    pass


class EmbeddingServiceError(APIError):
    """Embedding service errors."""

    pass


class CrawlServiceError(APIError):
    """Crawl service errors."""

    pass
