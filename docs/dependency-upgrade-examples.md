# Practical Implementation Examples

This document provides concrete examples of how to implement the dependency upgrades in the codebase.

## 1. Pydantic v2 Model Update Example

### Before (Current Code Pattern)
```python
from pydantic import BaseModel, validator

class EmbeddingConfig(BaseModel):
    model_name: str
    dimensions: int
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    class Config:
        extra = "forbid"
        
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v
        
    @validator('dimensions')
    def validate_dimensions(cls, v):
        if v not in [128, 256, 384, 512, 768, 1024, 1536]:
            raise ValueError(f'Unsupported dimensions: {v}')
        return v
```

### After (Pydantic v2 Pattern)
```python
from typing import Self
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, computed_field

class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        # New: better serialization control
        json_schema_serialization_defaults_required=True
    )
    
    model_name: str = Field(
        ..., 
        description="Name of the embedding model",
        examples=["text-embedding-3-small", "BAAI/bge-small-en-v1.5"]
    )
    dimensions: int = Field(
        ...,
        description="Vector dimensions",
        ge=1,
        le=4096
    )
    chunk_size: int = Field(
        default=512,
        description="Size of text chunks",
        ge=100,
        le=2000
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks",
        ge=0
    )
    
    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        supported = {128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096}
        if v not in supported:
            raise ValueError(f'Unsupported dimensions: {v}. Choose from: {sorted(supported)}')
        return v
    
    @model_validator(mode='after')
    def validate_chunk_relationship(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f'chunk_overlap ({self.chunk_overlap}) must be less than '
                f'chunk_size ({self.chunk_size})'
            )
        return self
    
    @computed_field
    @property
    def effective_chunk_size(self) -> int:
        """Calculate the effective chunk size after overlap."""
        return self.chunk_size - self.chunk_overlap
    
    @computed_field
    @property
    def cache_key(self) -> str:
        """Generate a cache key for this configuration."""
        return f"{self.model_name}:{self.dimensions}:{self.chunk_size}:{self.chunk_overlap}"
```

## 2. HTTPX Advanced Client Example

### Before (Basic Usage)
```python
import aiohttp

class DocumentFetcher:
    def __init__(self):
        self.session = None
        
    async def setup(self):
        self.session = aiohttp.ClientSession()
        
    async def fetch(self, url: str) -> str:
        async with self.session.get(url) as response:
            return await response.text()
            
    async def cleanup(self):
        if self.session:
            await self.session.close()
```

### After (HTTPX with Advanced Features)
```python
import httpx
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

class DocumentFetcher:
    def __init__(self, http2: bool = True):
        self.client: Optional[httpx.AsyncClient] = None
        self.http2 = http2
        
    async def __aenter__(self):
        # Configure advanced limits
        limits = httpx.Limits(
            max_keepalive_connections=50,
            max_connections=100,
            keepalive_expiry=300.0,  # 5 minutes
            max_keepalive_time=600.0,  # 10 minutes
        )
        
        # Configure timeouts
        timeout = httpx.Timeout(
            connect=5.0,
            read=30.0,
            write=10.0,
            pool=5.0
        )
        
        # Event hooks for monitoring
        async def log_request(request):
            logger.debug(f"Request: {request.method} {request.url}")
            
        async def add_metrics(response):
            metrics.record_http_request(
                method=response.request.method,
                url=str(response.request.url),
                status_code=response.status_code,
                duration=response.elapsed.total_seconds()
            )
        
        self.client = httpx.AsyncClient(
            http2=self.http2,
            limits=limits,
            timeout=timeout,
            follow_redirects=True,
            event_hooks={
                'request': [log_request],
                'response': [add_metrics]
            },
            headers={
                'User-Agent': 'AI-Docs-Scraper/1.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        reraise=True
    )
    async def fetch(self, url: str) -> str:
        """Fetch document with automatic retry logic."""
        response = await self.client.get(url)
        response.raise_for_status()
        return response.text
        
    async def fetch_many(self, urls: list[str]) -> list[str]:
        """Fetch multiple URLs concurrently with rate limiting."""
        # Use httpx's connection pooling for efficiency
        tasks = [self.fetch(url) for url in urls]
        
        # Modern asyncio pattern with TaskGroup
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self.fetch(url)) for url in urls]
            
        return [task.result() for task in tasks]
```

## 3. Modern Asyncio TaskGroup Example

### Before (Using gather)
```python
async def process_documents(docs: list[Document]) -> list[ProcessedDocument]:
    # Old pattern with gather
    tasks = []
    for doc in docs:
        tasks.append(process_single_document(doc))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions manually
    processed = []
    errors = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append((docs[i], result))
        else:
            processed.append(result)
            
    if errors:
        logger.error(f"Failed to process {len(errors)} documents")
        
    return processed
```

### After (Using TaskGroup with Exception Groups)
```python
async def process_documents(docs: list[Document]) -> list[ProcessedDocument]:
    processed = []
    failed_docs = []
    
    try:
        async with asyncio.TaskGroup() as tg:
            # Create tasks with better tracking
            tasks = {
                tg.create_task(
                    process_single_document(doc), 
                    name=f"process_doc_{doc.id}"
                ): doc 
                for doc in docs
            }
            
    except* ValidationError as eg:
        # Handle validation errors separately
        for error in eg.exceptions:
            task = next(t for t in tasks if t.exception() == error)
            doc = tasks[task]
            logger.warning(f"Validation failed for doc {doc.id}: {error}")
            failed_docs.append((doc, error))
            
    except* httpx.HTTPError as eg:
        # Handle HTTP errors separately
        for error in eg.exceptions:
            task = next(t for t in tasks if t.exception() == error)
            doc = tasks[task]
            logger.error(f"HTTP error for doc {doc.id}: {error}")
            failed_docs.append((doc, error))
            
    else:
        # All tasks completed successfully
        processed = [task.result() for task in tasks]
        
    return processed, failed_docs

# With semaphore for rate limiting
async def process_documents_with_limit(
    docs: list[Document], 
    max_concurrent: int = 10
) -> list[ProcessedDocument]:
    sem = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(doc: Document):
        async with sem:
            return await process_single_document(doc)
    
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(process_with_limit(doc))
            for doc in docs
        ]
        
    return [task.result() for task in tasks]
```

## 4. Enhanced Vector Search with Pydantic v2

```python
from typing import Annotated, Self
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from datetime import datetime

class VectorSearchRequest(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        str_strip_whitespace=True,
        # New: automatic validation of defaults
        validate_default=True,
        # New: better error messages
        use_attribute_docstrings=True
    )
    
    query: Annotated[str, Field(
        min_length=1,
        max_length=1000,
        description="Search query text"
    )]
    
    limit: Annotated[int, Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum results to return"
    )]
    
    threshold: Annotated[float, Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold"
    )]
    
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters"
    )
    
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in results"
    )
    
    search_type: Literal["semantic", "hybrid", "keyword"] = Field(
        default="hybrid",
        description="Type of search to perform"
    )
    
    @field_validator('query')
    @classmethod
    def clean_query(cls, v: str) -> str:
        # Remove extra whitespace
        v = ' '.join(v.split())
        # Remove dangerous characters
        v = re.sub(r'[<>{}]', '', v)
        return v
    
    @model_validator(mode='after')
    def validate_search_params(self) -> Self:
        # Adjust threshold based on search type
        if self.search_type == "keyword" and self.threshold < 0.5:
            self.threshold = 0.5
        return self
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for additional setup."""
        # Log search request for analytics
        logger.info(
            f"Search request: type={self.search_type}, "
            f"query_length={len(self.query)}, limit={self.limit}"
        )
```

## 5. Performance Optimized Embedding Manager

```python
from typing import AsyncGenerator
import numpy as np
from pydantic import BaseModel, computed_field
import asyncio

class EmbeddingBatch(BaseModel):
    texts: list[str]
    embeddings: np.ndarray
    model: str
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # New: custom serialization for numpy
        json_encoders={
            np.ndarray: lambda v: v.tolist()
        }
    )
    
    @computed_field
    @property
    def batch_size(self) -> int:
        return len(self.texts)
    
    @computed_field
    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 0

class OptimizedEmbeddingManager:
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self.batch_queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
        self.processing = False
        
    async def get_embedding(self, text: str) -> np.ndarray:
        """Queue single text for batch processing."""
        future = asyncio.Future()
        await self.batch_queue.put((text, future))
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
            
        return await future
        
    async def _process_batch(self):
        """Process embeddings in batches for efficiency."""
        self.processing = True
        batch_size = 100
        batch_timeout = 0.1  # 100ms
        
        try:
            while True:
                batch = []
                futures = []
                
                # Collect batch with timeout
                deadline = asyncio.get_event_loop().time() + batch_timeout
                
                while len(batch) < batch_size:
                    try:
                        remaining = max(0, deadline - asyncio.get_event_loop().time())
                        text, future = await asyncio.wait_for(
                            self.batch_queue.get(), 
                            timeout=remaining
                        )
                        batch.append(text)
                        futures.append(future)
                    except asyncio.TimeoutError:
                        break
                        
                if not batch:
                    # No items in queue
                    self.processing = False
                    break
                    
                # Process batch
                try:
                    embeddings = await self.provider.embed_batch(batch)
                    
                    # Distribute results
                    for future, embedding in zip(futures, embeddings):
                        future.set_result(embedding)
                        
                except Exception as e:
                    # Propagate error to all futures
                    for future in futures:
                        future.set_exception(e)
                        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self.processing = False
```

## 6. Redis Cache with Modern Patterns

```python
import redis.asyncio as redis
from typing import Optional, TypeVar, Generic
from pydantic import BaseModel
import json

T = TypeVar('T', bound=BaseModel)

class ModernRedisCache(Generic[T]):
    def __init__(
        self, 
        redis_url: str,
        model_class: type[T],
        key_prefix: str = "cache",
        ttl: int = 3600
    ):
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=50,
            health_check_interval=30,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 2,  # TCP_KEEPINTVL  
                3: 3,  # TCP_KEEPCNT
            }
        )
        self.redis = redis.Redis(connection_pool=self.pool)
        self.model_class = model_class
        self.key_prefix = key_prefix
        self.ttl = ttl
        
    async def get(self, key: str) -> Optional[T]:
        """Get item from cache with automatic deserialization."""
        full_key = f"{self.key_prefix}:{key}"
        
        try:
            data = await self.redis.get(full_key)
            if data:
                # Use Pydantic v2's model_validate_json
                return self.model_class.model_validate_json(data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            
        return None
        
    async def set(self, key: str, value: T) -> bool:
        """Set item in cache with automatic serialization."""
        full_key = f"{self.key_prefix}:{key}"
        
        try:
            # Use Pydantic v2's model_dump_json
            data = value.model_dump_json(
                exclude_unset=True,
                exclude_defaults=False
            )
            await self.redis.set(full_key, data, ex=self.ttl)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
            
    async def get_many(self, keys: list[str]) -> dict[str, T]:
        """Get multiple items efficiently with pipeline."""
        full_keys = [f"{self.key_prefix}:{key}" for key in keys]
        
        async with self.redis.pipeline() as pipe:
            for full_key in full_keys:
                pipe.get(full_key)
            results = await pipe.execute()
            
        items = {}
        for key, data in zip(keys, results):
            if data:
                try:
                    items[key] = self.model_class.model_validate_json(data)
                except Exception as e:
                    logger.error(f"Failed to deserialize {key}: {e}")
                    
        return items
```

These examples demonstrate practical implementations of the new features available in the updated dependencies, focusing on real-world use cases in the AI documentation vector database system.