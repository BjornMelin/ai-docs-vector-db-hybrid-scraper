"""Protocol definitions for sophisticated dependency injection.

This module defines the protocol interfaces that enable clean 
dependency injection patterns and sophisticated component design
while maintaining flexibility and testability.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from abc import abstractmethod
from contextlib import AsyncContextManager

from .types import (
    SearchOptions,
    EmbeddingOptions, 
    ProcessingOptions,
    SearchResult,
    EmbeddingVector,
    DocumentId,
    MetadataDict,
    ProgressiveResponse,
)


@runtime_checkable
class SearchProtocol(Protocol):
    """Protocol for search implementations.
    
    This protocol defines the interface for search providers,
    allowing for pluggable search implementations while maintaining
    type safety and consistent behavior.
    """
    
    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        options: Optional[SearchOptions] = None,
    ) -> List[SearchResult]:
        """Perform a search query.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            options: Advanced search options
            
        Returns:
            List of search results
        """
        ...
    
    @abstractmethod
    async def add_document(
        self,
        content: str,
        *,
        metadata: Optional[MetadataDict] = None,
    ) -> DocumentId:
        """Add a document to the search index.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Document identifier
        """
        ...
    
    @abstractmethod
    async def update_document(
        self,
        document_id: DocumentId,
        *,
        content: Optional[str] = None,
        metadata: Optional[MetadataDict] = None,
    ) -> bool:
        """Update an existing document.
        
        Args:
            document_id: Document identifier
            content: New content (optional)
            metadata: New metadata (optional)
            
        Returns:
            Success status
        """
        ...
    
    @abstractmethod
    async def delete_document(self, document_id: DocumentId) -> bool:
        """Delete a document from the search index.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Success status
        """
        ...
    
    @abstractmethod
    async def get_document(self, document_id: DocumentId) -> Optional[SearchResult]:
        """Retrieve a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        ...
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get search provider statistics.
        
        Returns:
            Dictionary with statistics
        """
        ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for embedding implementations.
    
    This protocol defines the interface for embedding providers,
    supporting multiple providers with consistent behavior.
    """
    
    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
        *,
        options: Optional[EmbeddingOptions] = None,
    ) -> EmbeddingVector:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            options: Embedding options
            
        Returns:
            Embedding vector
        """
        ...
    
    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        *,
        options: Optional[EmbeddingOptions] = None,
    ) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            options: Embedding options
            
        Returns:
            List of embedding vectors
        """
        ...
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension
        """
        ...
    
    @abstractmethod
    async def estimate_cost(
        self,
        texts: List[str],
        *,
        options: Optional[EmbeddingOptions] = None,
    ) -> float:
        """Estimate the cost of generating embeddings.
        
        Args:
            texts: List of input texts
            options: Embedding options
            
        Returns:
            Estimated cost in USD
        """
        ...
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the embedding provider.
        
        Returns:
            Provider information dictionary
        """
        ...


@runtime_checkable  
class DocumentProcessorProtocol(Protocol):
    """Protocol for document processing implementations.
    
    This protocol defines the interface for document processors,
    enabling pluggable processing pipelines.
    """
    
    @abstractmethod
    async def process_document(
        self,
        content: str,
        *,
        content_type: str = "text",
        options: Optional[ProcessingOptions] = None,
    ) -> Dict[str, Any]:
        """Process a document and extract structured information.
        
        Args:
            content: Document content
            content_type: Type of content
            options: Processing options
            
        Returns:
            Processed document data
        """
        ...
    
    @abstractmethod
    async def chunk_document(
        self,
        content: str,
        *,
        chunk_size: int = 1000,
        overlap: int = 200,
        options: Optional[ProcessingOptions] = None,
    ) -> List[Dict[str, Any]]:
        """Split document into chunks.
        
        Args:
            content: Document content
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
            options: Processing options
            
        Returns:
            List of document chunks
        """
        ...
    
    @abstractmethod
    async def extract_metadata(
        self,
        content: str,
        *,
        content_type: str = "text",
    ) -> MetadataDict:
        """Extract metadata from document content.
        
        Args:
            content: Document content
            content_type: Type of content
            
        Returns:
            Extracted metadata
        """
        ...
    
    @abstractmethod
    async def analyze_content(
        self,
        content: str,
        *,
        include_sentiment: bool = False,
        include_topics: bool = False,
        include_entities: bool = False,
    ) -> Dict[str, Any]:
        """Analyze document content for insights.
        
        Args:
            content: Document content
            include_sentiment: Include sentiment analysis
            include_topics: Include topic extraction
            include_entities: Include entity extraction
            
        Returns:
            Content analysis results
        """
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations.
    
    This protocol defines the interface for cache providers,
    supporting multiple caching strategies.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        ...
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        ...
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cached values.
        
        Returns:
            Success status
        """
        ...
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        ...


@runtime_checkable
class CrawlerProtocol(Protocol):
    """Protocol for web crawling implementations.
    
    This protocol defines the interface for web crawlers,
    supporting multiple crawling strategies.
    """
    
    @abstractmethod
    async def crawl_url(
        self,
        url: str,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Crawl a single URL.
        
        Args:
            url: URL to crawl
            options: Crawling options
            
        Returns:
            Crawled content and metadata
        """
        ...
    
    @abstractmethod
    async def crawl_urls(
        self,
        urls: List[str],
        *,
        batch_size: int = 10,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Crawl multiple URLs.
        
        Args:
            urls: List of URLs to crawl
            batch_size: Number of URLs to crawl concurrently
            options: Crawling options
            
        Returns:
            List of crawled content and metadata
        """
        ...
    
    @abstractmethod
    async def get_sitemap(self, base_url: str) -> List[str]:
        """Extract URLs from sitemap.
        
        Args:
            base_url: Base URL to find sitemap
            
        Returns:
            List of URLs from sitemap
        """
        ...


@runtime_checkable
class MonitoringProtocol(Protocol):
    """Protocol for monitoring and observability implementations.
    
    This protocol defines the interface for monitoring providers.
    """
    
    @abstractmethod
    async def record_metric(
        self,
        name: str,
        value: float,
        *,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        ...
    
    @abstractmethod
    async def increment_counter(
        self,
        name: str,
        *,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags
        """
        ...
    
    @abstractmethod
    async def record_timing(
        self,
        name: str,
        duration_ms: float,
        *,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a timing metric.
        
        Args:
            name: Timing metric name
            duration_ms: Duration in milliseconds
            tags: Optional tags
        """
        ...
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics.
        
        Returns:
            Dictionary of metrics
        """
        ...


@runtime_checkable
class ComponentLifecycle(Protocol):
    """Protocol for component lifecycle management.
    
    This protocol defines initialization and cleanup patterns.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        ...
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup component resources."""
        ...
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        ...


@runtime_checkable
class HealthCheckProtocol(Protocol):
    """Protocol for health check implementations."""
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status information
        """
        ...
    
    @abstractmethod
    async def readiness_check(self) -> bool:
        """Check if component is ready to serve requests.
        
        Returns:
            Readiness status
        """
        ...
    
    @abstractmethod
    async def liveness_check(self) -> bool:
        """Check if component is alive and functioning.
        
        Returns:
            Liveness status
        """
        ...


# Composite protocols for sophisticated dependency injection

@runtime_checkable
class AdvancedSearchProvider(SearchProtocol, ComponentLifecycle, HealthCheckProtocol, Protocol):
    """Advanced search provider with lifecycle and health checking."""
    pass


@runtime_checkable
class ManagedEmbeddingProvider(EmbeddingProtocol, ComponentLifecycle, MonitoringProtocol, Protocol):
    """Managed embedding provider with monitoring capabilities."""
    pass


@runtime_checkable
class ObservableCache(CacheProtocol, MonitoringProtocol, HealthCheckProtocol, Protocol):
    """Observable cache with monitoring and health checking."""
    pass


# Context manager protocols for resource management

@runtime_checkable
class AsyncContextProvider(Protocol):
    """Protocol for async context managers."""
    
    def __aenter__(self) -> AsyncContextManager[Any]:
        """Async context manager entry."""
        ...
    
    def __aexit__(self, exc_type, exc_val, exc_tb) -> AsyncContextManager[None]:
        """Async context manager exit."""
        ...