# Code Architecture Improvements

> **Status:** Specifications Complete  
> **Priority:** High  
> **Estimated Effort:** 2-3 weeks  
> **Documentation Created:** 2025-05-22

## Overview

Comprehensive refactoring to eliminate code duplication, improve maintainability, and create a clean, scalable architecture. This addresses technical debt accumulated during rapid feature development and establishes patterns for future growth.

## Current Architecture Issues

### Code Duplication Problems
- **Embedding Logic**: Duplicated across multiple modules
- **Qdrant Initialization**: Repeated client setup code
- **Configuration Loading**: Scattered configuration patterns
- **Error Handling**: Inconsistent error patterns
- **Validation**: Repeated validation logic

### Module Organization Issues
- **Flat Structure**: All modules in src/ without clear separation
- **Mixed Concerns**: Business logic mixed with infrastructure
- **Tight Coupling**: Direct dependencies between unrelated modules
- **No Clear Interfaces**: Missing abstraction layers

## Target Architecture

### Clean Architecture Principles
```
src/
├── core/           # Core business logic (no external dependencies)
├── services/       # Business services and orchestration
├── providers/      # External API integrations
├── infrastructure/ # Framework and external concerns
├── models/         # Data models and schemas
├── exceptions/     # Custom exception hierarchy
└── utils/          # Shared utilities
```

### Dependency Flow
```
Infrastructure → Services → Core
     ↑              ↑        ↑
Providers      Models   Exceptions
     ↑              ↑        ↑
   Utils       Utils    Utils
```

## Implementation Plan

### Phase 1: Core Domain Models

#### Unified Data Models
```python
# src/models/document.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any

class DocumentMetadata(BaseModel):
    """Document metadata schema."""
    url: str
    title: Optional[str] = None
    language: Optional[str] = None
    content_type: str = Field(default="text/html")
    crawl_timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_provider: str = Field(default="crawl4ai")
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)

class DocumentChunk(BaseModel):
    """Document chunk with embedding."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None
    metadata: DocumentMetadata

class SearchResult(BaseModel):
    """Search result with scoring."""
    chunk: DocumentChunk
    score: float
    rank: int
    relevance_score: Optional[float] = None
    rerank_score: Optional[float] = None
```

#### Configuration Models
```python
# src/models/config.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal

class QdrantConfig(BaseModel):
    """Qdrant configuration."""
    url: str = Field(default="http://localhost:6333")
    api_key: Optional[str] = None
    timeout: float = Field(default=30.0, gt=0)
    prefer_grpc: bool = Field(default=False)
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""
    provider: Literal["openai", "fastembed", "auto"] = "auto"
    model: str = Field(default="text-embedding-3-small")
    dimensions: Optional[int] = None
    batch_size: int = Field(default=100, gt=0, le=2048)
    api_key: Optional[str] = None
    
class UnifiedConfig(BaseModel):
    """Complete application configuration."""
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # Performance settings
    max_concurrent_requests: int = Field(default=10, gt=0, le=100)
    chunk_size: int = Field(default=1600, gt=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)
    
    # Cache settings
    enable_cache: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, gt=0)
    
    # Quality settings
    enable_reranking: bool = Field(default=True)
    rerank_top_k: int = Field(default=50, gt=0)
```

### Phase 2: Service Layer Architecture

#### Core Services Interface
```python
# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.models.document import DocumentChunk, SearchResult

class EmbeddingService(ABC):
    """Abstract embedding service."""
    
    @abstractmethod
    async def generate_embeddings(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensions."""
        pass

class VectorService(ABC):
    """Abstract vector database service."""
    
    @abstractmethod
    async def create_collection(
        self, 
        name: str, 
        vector_size: int
    ) -> bool:
        """Create vector collection."""
        pass
    
    @abstractmethod
    async def upsert_chunks(
        self, 
        collection: str, 
        chunks: List[DocumentChunk]
    ) -> bool:
        """Store document chunks."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        collection: str, 
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

class CrawlService(ABC):
    """Abstract crawling service."""
    
    @abstractmethod
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape single URL."""
        pass
```

#### Centralized Client Manager
```python
# src/infrastructure/client_manager.py
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncio
from qdrant_client import QdrantClient
from openai import AsyncOpenAI

class ClientManager:
    """Centralized API client management."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._clients: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
    
    async def get_qdrant_client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if "qdrant" not in self._clients:
            if "qdrant" not in self._locks:
                self._locks["qdrant"] = asyncio.Lock()
            
            async with self._locks["qdrant"]:
                if "qdrant" not in self._clients:
                    self._clients["qdrant"] = QdrantClient(
                        url=self.config.qdrant.url,
                        api_key=self.config.qdrant.api_key,
                        timeout=self.config.qdrant.timeout,
                        prefer_grpc=self.config.qdrant.prefer_grpc
                    )
        
        return self._clients["qdrant"]
    
    async def get_openai_client(self) -> Optional[AsyncOpenAI]:
        """Get or create OpenAI client."""
        if not self.config.embedding.api_key:
            return None
            
        if "openai" not in self._clients:
            if "openai" not in self._locks:
                self._locks["openai"] = asyncio.Lock()
            
            async with self._locks["openai"]:
                if "openai" not in self._clients:
                    self._clients["openai"] = AsyncOpenAI(
                        api_key=self.config.embedding.api_key
                    )
        
        return self._clients["openai"]
    
    @asynccontextmanager
    async def health_check(self):
        """Check client health and auto-reconnect."""
        try:
            # Health check logic here
            yield
        except Exception as e:
            # Invalidate clients on error
            self._clients.clear()
            raise
    
    async def close(self):
        """Close all clients."""
        for client in self._clients.values():
            if hasattr(client, 'close'):
                await client.close()
        self._clients.clear()
```

#### Service Implementations
```python
# src/services/embedding_service.py
from src.core.interfaces import EmbeddingService
from src.infrastructure.client_manager import ClientManager
from src.providers.openai_provider import OpenAIEmbeddingProvider
from src.providers.fastembed_provider import FastEmbedProvider

class UnifiedEmbeddingService(EmbeddingService):
    """Unified embedding service with provider abstraction."""
    
    def __init__(self, client_manager: ClientManager, config: EmbeddingConfig):
        self.client_manager = client_manager
        self.config = config
        self._provider = None
    
    async def _get_provider(self) -> EmbeddingService:
        """Get appropriate embedding provider."""
        if self._provider is None:
            if self.config.provider == "openai":
                client = await self.client_manager.get_openai_client()
                if client:
                    self._provider = OpenAIEmbeddingProvider(client, self.config)
                else:
                    raise ValueError("OpenAI API key required for OpenAI provider")
            elif self.config.provider == "fastembed":
                self._provider = FastEmbedProvider(self.config)
            else:  # auto
                # Smart selection logic
                client = await self.client_manager.get_openai_client()
                if client:
                    self._provider = OpenAIEmbeddingProvider(client, self.config)
                else:
                    self._provider = FastEmbedProvider(self.config)
        
        return self._provider
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using selected provider."""
        provider = await self._get_provider()
        return await provider.generate_embeddings(texts)
    
    @property
    def dimensions(self) -> int:
        """Get vector dimensions."""
        if self._provider:
            return self._provider.dimensions
        
        # Default dimensions based on config
        if self.config.provider == "openai":
            return self.config.dimensions or 1536
        else:
            return 384  # FastEmbed default
```

### Phase 3: Provider Abstraction Layer

#### Provider Implementations
```python
# src/providers/openai_provider.py
from openai import AsyncOpenAI
from src.core.interfaces import EmbeddingService
from src.exceptions.embedding_exceptions import EmbeddingError

class OpenAIEmbeddingProvider(EmbeddingService):
    """OpenAI embedding provider."""
    
    def __init__(self, client: AsyncOpenAI, config: EmbeddingConfig):
        self.client = client
        self.config = config
        self._dimensions = config.dimensions or 1536
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate OpenAI embeddings with batching."""
        if not texts:
            return []
        
        try:
            all_embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.config.model,
                    dimensions=self._dimensions
                )
                
                batch_embeddings = [
                    embedding.embedding 
                    for embedding in response.data
                ]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding generation failed: {e}")
    
    @property
    def dimensions(self) -> int:
        return self._dimensions

# src/providers/fastembed_provider.py
from fastembed import TextEmbedding
from src.core.interfaces import EmbeddingService
from src.exceptions.embedding_exceptions import EmbeddingError

class FastEmbedProvider(EmbeddingService):
    """FastEmbed local provider."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        try:
            self.model = TextEmbedding(config.model)
            self._dimensions = self.model.dim
        except Exception as e:
            raise EmbeddingError(f"FastEmbed initialization failed: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate local embeddings."""
        if not texts:
            return []
        
        try:
            embeddings = list(self.model.embed(texts))
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise EmbeddingError(f"FastEmbed generation failed: {e}")
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
```

### Phase 4: Shared Utilities and Patterns

#### Common Decorators
```python
# src/utils/decorators.py
import asyncio
import functools
from typing import TypeVar, Callable, Any
from src.exceptions.base_exceptions import RetryableError

T = TypeVar('T')

def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """Async retry decorator with exponential backoff."""
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except RetryableError as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break
                    
                    delay = min(
                        base_delay * (backoff_factor ** attempt),
                        max_delay
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable error
                    raise e
            
            raise last_exception
        
        return wrapper
    return decorator

def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Circuit breaker pattern decorator."""
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        failure_count = 0
        last_failure_time = None
        state = "closed"  # closed, open, half-open
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, state
            
            if state == "open":
                if (
                    last_failure_time and 
                    time.time() - last_failure_time > recovery_timeout
                ):
                    state = "half-open"
                else:
                    raise CircuitBreakerError("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                
                # Reset on success
                if state == "half-open":
                    state = "closed"
                    failure_count = 0
                
                return result
                
            except Exception as e:
                failure_count += 1
                last_failure_time = time.time()
                
                if failure_count >= failure_threshold:
                    state = "open"
                
                raise e
        
        return wrapper
    return decorator
```

#### Configuration Loading Utilities
```python
# src/utils/config_loader.py
import os
from pathlib import Path
from typing import Type, TypeVar, Dict, Any
from pydantic import BaseModel
import yaml
import json

T = TypeVar('T', bound=BaseModel)

class ConfigLoader:
    """Centralized configuration loading."""
    
    @staticmethod
    def load_from_env(config_class: Type[T]) -> T:
        """Load configuration from environment variables."""
        return config_class()
    
    @staticmethod
    def load_from_file(config_class: Type[T], file_path: Path) -> T:
        """Load configuration from file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
        
        return config_class(**data)
    
    @staticmethod
    def merge_configs(*configs: BaseModel) -> BaseModel:
        """Merge multiple configurations with precedence."""
        merged_data = {}
        
        for config in configs:
            merged_data.update(config.dict())
        
        # Return the same type as first config
        return type(configs[0])(**merged_data)
    
    @staticmethod
    def create_unified_config(
        config_file: Path | None = None
    ) -> UnifiedConfig:
        """Create unified configuration from multiple sources."""
        
        # Load from environment (lowest precedence)
        env_config = ConfigLoader.load_from_env(UnifiedConfig)
        
        # Load from file if provided (highest precedence)
        if config_file and config_file.exists():
            file_config = ConfigLoader.load_from_file(UnifiedConfig, config_file)
            return ConfigLoader.merge_configs(env_config, file_config)
        
        return env_config
```

### Phase 5: Exception Hierarchy

#### Structured Exception System
```python
# src/exceptions/base_exceptions.py
class ApplicationError(Exception):
    """Base application exception."""
    
    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code
        super().__init__(message)

class RetryableError(ApplicationError):
    """Exception that can be retried."""
    pass

class ConfigurationError(ApplicationError):
    """Configuration-related errors."""
    pass

class ValidationError(ApplicationError):
    """Data validation errors."""
    pass

# src/exceptions/embedding_exceptions.py
class EmbeddingError(ApplicationError):
    """Embedding service errors."""
    pass

class EmbeddingProviderError(EmbeddingError, RetryableError):
    """Provider-specific errors that can be retried."""
    pass

class EmbeddingQuotaError(EmbeddingError):
    """Quota/rate limit errors."""
    pass

# src/exceptions/vector_exceptions.py
class VectorDatabaseError(ApplicationError):
    """Vector database errors."""
    pass

class CollectionNotFoundError(VectorDatabaseError):
    """Collection doesn't exist."""
    pass

class VectorInsertError(VectorDatabaseError, RetryableError):
    """Vector insertion failures."""
    pass
```

## Refactoring Strategy

### Phase 1: Models and Interfaces (Week 1)
1. Create new model structure in `src/models/`
2. Define core interfaces in `src/core/`
3. Update existing code to use new models
4. Comprehensive testing of models

### Phase 2: Service Layer (Week 1-2)
1. Implement client manager
2. Create service implementations
3. Update MCP server to use new services
4. Integration testing

### Phase 3: Provider Abstraction (Week 2)
1. Implement provider classes
2. Create provider factory
3. Update embedding and crawling logic
4. Provider-specific testing

### Phase 4: Utilities and Cleanup (Week 2-3)
1. Implement shared utilities
2. Add decorator patterns
3. Remove duplicate code
4. Final integration testing

### Phase 5: Documentation and Migration (Week 3)
1. Update all documentation
2. Create migration guides
3. Performance benchmarking
4. Final code review

## Code Quality Improvements

### Type Safety Enhancements
```python
# Strict typing throughout
from typing import TypeVar, Generic, Protocol, runtime_checkable

@runtime_checkable
class Embeddable(Protocol):
    """Protocol for embeddable content."""
    
    def get_text(self) -> str:
        """Get text content for embedding."""
        ...

T = TypeVar('T', bound=Embeddable)

class DocumentProcessor(Generic[T]):
    """Generic document processor."""
    
    async def process(self, items: List[T]) -> List[DocumentChunk]:
        """Process embeddable items into chunks."""
        chunks = []
        for item in items:
            text = item.get_text()
            # Processing logic here
        return chunks
```

### Testing Improvements
```python
# src/tests/conftest.py
import pytest
from unittest.mock import AsyncMock
from src.models.config import UnifiedConfig
from src.infrastructure.client_manager import ClientManager

@pytest.fixture
def mock_config():
    """Test configuration."""
    return UnifiedConfig(
        qdrant=QdrantConfig(url="http://test:6333"),
        embedding=EmbeddingConfig(provider="fastembed")
    )

@pytest.fixture
async def client_manager(mock_config):
    """Test client manager."""
    manager = ClientManager(mock_config)
    yield manager
    await manager.close()

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = AsyncMock()
    service.generate_embeddings.return_value = [[0.1] * 384]
    service.dimensions = 384
    return service
```

## Migration Tools

### Code Migration Script
```python
#!/usr/bin/env python3
"""Migration script for architecture refactor."""

import ast
import re
from pathlib import Path
from typing import List, Dict

class CodeMigrator:
    """Migrate code to new architecture."""
    
    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.migrations = []
    
    def add_migration(self, pattern: str, replacement: str):
        """Add regex migration rule."""
        self.migrations.append((re.compile(pattern), replacement))
    
    def migrate_file(self, file_path: Path) -> str:
        """Migrate single file."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply migrations
        for pattern, replacement in self.migrations:
            content = pattern.sub(replacement, content)
        
        return content
    
    def migrate_all(self):
        """Migrate all Python files."""
        for py_file in self.src_dir.glob("**/*.py"):
            migrated = self.migrate_file(py_file)
            
            with open(py_file, 'w') as f:
                f.write(migrated)

# Migration rules
migrator = CodeMigrator(Path("src"))

# Update imports
migrator.add_migration(
    r"from src\.chunking import",
    "from src.services.chunking_service import"
)

migrator.add_migration(
    r"from src\.crawl4ai_bulk_embedder import",
    "from src.services.document_service import"
)

# Update class names
migrator.add_migration(
    r"QdrantClient\(",
    "await client_manager.get_qdrant_client()"
)

migrator.migrate_all()
```

## Performance Monitoring

### Architecture Metrics
```python
# src/utils/metrics.py
from dataclasses import dataclass
from typing import Dict, Any
import time

@dataclass
class PerformanceMetrics:
    """Performance tracking."""
    operation: str
    duration: float
    memory_usage: int
    success: bool
    metadata: Dict[str, Any]

class MetricsCollector:
    """Collect architecture performance metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    async def track_operation(self, operation: str, func, *args, **kwargs):
        """Track operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise
        finally:
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            metric = PerformanceMetrics(
                operation=operation,
                duration=duration,
                memory_usage=memory_usage,
                success=success,
                metadata={"args_count": len(args)}
            )
            self.metrics.append(metric)
        
        return result
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
```

## Official Documentation References

### Python Architecture
- **Clean Architecture**: https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html
- **Dependency Injection**: https://python-dependency-injector.ets-labs.org/
- **Pydantic**: https://docs.pydantic.dev/latest/
- **AsyncIO**: https://docs.python.org/3/library/asyncio.html

### Design Patterns
- **Repository Pattern**: https://martinfowler.com/eaaCatalog/repository.html
- **Circuit Breaker**: https://martinfowler.com/bliki/CircuitBreaker.html
- **Retry Pattern**: https://docs.microsoft.com/en-us/azure/architecture/patterns/retry
- **Singleton Pattern**: https://refactoring.guru/design-patterns/singleton/python/example

### Testing
- **pytest-asyncio**: https://pytest-asyncio.readthedocs.io/
- **unittest.mock**: https://docs.python.org/3/library/unittest.mock.html
- **pytest fixtures**: https://docs.pytest.org/en/stable/fixture.html

## Success Criteria

### Code Quality Metrics
- [ ] Eliminate 90%+ code duplication
- [ ] Achieve 95%+ test coverage
- [ ] Zero circular dependencies
- [ ] All modules follow single responsibility

### Architecture Metrics
- [ ] Clear separation of concerns
- [ ] Consistent error handling patterns
- [ ] Configurable dependency injection
- [ ] Maintainable abstractions

### Performance Metrics
- [ ] No performance regression
- [ ] 20%+ reduction in memory usage
- [ ] Faster startup time
- [ ] Improved error recovery

This architecture refactor will create a maintainable, scalable foundation for all future development.