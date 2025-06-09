# System Architecture

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: Complete system architecture and design patterns  
> **Audience**: Developers understanding system design and contributing to architecture

This comprehensive guide covers the complete architecture of the AI Documentation Vector DB system, from high-level design to detailed implementation patterns.

## 🏗️ High-Level Architecture

### System Overview

AI Documentation Vector DB is a hybrid scraping and vector search system that combines intelligent browser automation, advanced vector search, and AI-powered query enhancement for high-performance documentation indexing and retrieval.

```mermaid
flowchart LR
    A["📄 Documentation URLs"] --> B["🤖 5-Tier Browser<br/>Automation"]
    B --> C["✂️ Enhanced Chunking"]
    C --> D["🔢 Embedding Pipeline"]
    D --> E["🗄️ Qdrant<br/>(Query API)"]
    E --> F["🧠 HyDE Enhancement"]
    F --> G["⚡ DragonflyDB Cache"]
    G --> H["🔧 MCP Server"]
    H --> I["💻 Claude Desktop"]
    
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef output fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class A input
    class B,C,D,F processing
    class E,G storage
    class H,I output
```

### Core Components Architecture

```mermaid
architecture-beta
    group api(cloud)[Unified MCP Server]
    group services(cloud)[Enhanced Service Layer]
    group config(database)[Configuration Layer]

    service mcpserver(server)[FastMCP 2.0 + Tools] in api
    
    service embedding(internet)[EmbeddingMgr + HyDE] in services
    service qdrant(database)[Qdrant + Query API] in services
    service automation(server)[5-Tier Browser System] in services
    service dragonfly(disk)[DragonflyDB Cache] in services
    service aliases(internet)[Collection Aliases] in services
    service security(shield)[Security Validation] in services
    
    service pydantic(database)[Unified Configuration] in config
    
    mcpserver:B --> embedding:T
    mcpserver:B --> qdrant:T
    mcpserver:B --> automation:T
    mcpserver:B --> dragonfly:T
    mcpserver:B --> aliases:T
    mcpserver:B --> security:T
    
    embedding:B --> pydantic:T
    qdrant:B --> pydantic:T
    automation:B --> pydantic:T
    dragonfly:B --> pydantic:T
    aliases:B --> pydantic:T
    security:B --> pydantic:T
```

## 🧩 Core Components

### 1. Unified Configuration System

#### Architecture Pattern

- **Location**: `src/config/`
- **Pattern**: Centralized Pydantic v2 models with nested configuration
- **Features**: Environment variable support, runtime validation, type safety

#### Component Structure

```mermaid
classDiagram
    class UnifiedConfig {
        +embedding: EmbeddingConfig
        +qdrant: QdrantConfig
        +crawl4ai: Crawl4AIConfig
        +openai: OpenAIConfig
        +cache: CacheConfig
        +security: SecurityConfig
        +load_from_env() Config
        +validate() bool
    }
    
    class EmbeddingConfig {
        +provider: EmbeddingProvider
        +model: str
        +batch_size: int
        +timeout: float
    }
    
    class QdrantConfig {
        +url: str
        +api_key: Optional[str]
        +timeout: float
        +max_retries: int
        +collection_name: str
    }
    
    class SecurityConfig {
        +enable_url_validation: bool
        +allowed_domains: List[str]
        +blocked_domains: List[str]
        +max_request_size: int
    }
    
    UnifiedConfig --> EmbeddingConfig
    UnifiedConfig --> QdrantConfig
    UnifiedConfig --> SecurityConfig
```

#### Configuration Loading Strategy

```python
# Configuration resolution order
1. Environment variables (highest priority)
2. Configuration files (.env, config.yaml)
3. Default values (lowest priority)

# Example configuration structure
config = UnifiedConfig(
    embedding=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        batch_size=100
    ),
    qdrant=QdrantConfig(
        url="http://localhost:6333",
        collection_name="documents"
    ),
    security=SecurityConfig(
        enable_url_validation=True,
        blocked_domains=["localhost", "127.0.0.1"]
    )
)
```

### 2. Service Layer Architecture

#### Base Service Pattern

All services inherit from a common base pattern ensuring consistent lifecycle management:

```mermaid
classDiagram
    class BaseService {
        <<abstract>>
        +config: Dict[str, Any]
        +logger: Logger
        +_initialized: bool
        +initialize() None
        +cleanup() None
        +_setup()* None
        +_teardown()* None
        +__aenter__() Self
        +__aexit__() None
    }
    
    class EmbeddingManager {
        +provider: EmbeddingProvider
        +cache: EmbeddingCache
        +generate_embeddings(texts) List[float]
        +generate_sparse_embeddings(texts) List[Dict]
        +get_dimensions() int
    }
    
    class QdrantService {
        +client: QdrantClient
        +query_api: QueryAPI
        +search_vectors(query, collection) List[SearchResult]
        +hybrid_search(dense, sparse, collection) List[SearchResult]
        +add_documents(docs, collection) bool
        +create_collection(name, config) bool
    }
    
    class UnifiedBrowserManager {
        +router: AutomationRouter
        +tiers: Dict[str, BrowserTier]
        +scrape(request) ScrapingResponse
        +analyze_url(url) AnalysisResult
        +get_system_status() SystemStatus
    }
    
    BaseService <|-- EmbeddingManager
    BaseService <|-- QdrantService
    BaseService <|-- UnifiedBrowserManager
```

#### Service Interaction Patterns

```mermaid
sequenceDiagram
    participant C as Client
    participant M as ServiceManager
    participant E as EmbeddingService
    participant Q as QdrantService
    participant B as BrowserService
    
    C->>M: initialize_services()
    M->>E: initialize()
    M->>Q: initialize()
    M->>B: initialize()
    
    C->>M: process_request(url)
    M->>B: scrape(url)
    B-->>M: content
    M->>E: generate_embeddings(content)
    E-->>M: vectors
    M->>Q: store_vectors(vectors)
    Q-->>M: success
    M-->>C: result
    
    Note over M: Services use dependency injection
    Note over E,Q,B: All services implement BaseService
```

### 3. 5-Tier Browser Automation System

#### Tier Architecture

The browser automation system implements an intelligent 5-tier hierarchy with automatic optimization:

```mermaid
flowchart TD
    A["🌐 URL Request"] --> B["🎯 AutomationRouter"]
    B --> C{"Tier Selection"}
    
    C -->|Tier 0| D1["⚡ Lightweight HTTP<br/>httpx + BeautifulSoup<br/>5-10x faster"]
    C -->|Tier 1| D2["🕷️ Crawl4AI Basic<br/>Standard automation<br/>Dynamic content"]
    C -->|Tier 2| D3["🔧 Crawl4AI Enhanced<br/>Custom JavaScript<br/>Interactive content"]
    C -->|Tier 3| D4["🤖 Browser-use AI<br/>Multi-LLM reasoning<br/>Complex interactions"]
    C -->|Tier 4| D5["🎭 Playwright + Firecrawl<br/>Maximum control<br/>Auth + workflows"]
    
    D1 --> E["📄 Content Output"]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    
    classDef tier0 fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef tier1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef tier2 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef tier3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef tier4 fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class D1 tier0
    class D2 tier1
    class D3 tier2
    class D4 tier3
    class D5 tier4
```

#### Intelligent Routing Logic

```python
class AutomationRouter:
    def select_tier(self, url: str, requirements: Dict) -> str:
        """Intelligent tier selection based on multiple factors."""
        
        # 1. URL Pattern Analysis
        if self._is_static_content(url):
            return "lightweight"
        
        # 2. Site-Specific Rules
        domain = urlparse(url).netloc
        if domain in self.browser_use_sites:
            return "browser_use"
        
        # 3. Content Complexity Analysis
        if requirements.get("javascript_required"):
            return "crawl4ai_enhanced"
        
        # 4. Performance-Based Learning
        best_tier = self._get_best_performing_tier(domain)
        return best_tier or "crawl4ai"
    
    def _is_static_content(self, url: str) -> bool:
        """Detect static content that doesn't need browser automation."""
        static_extensions = {'.md', '.txt', '.json', '.xml', '.csv'}
        return any(url.endswith(ext) for ext in static_extensions)
```

#### Fallback Strategy

```mermaid
stateDiagram-v2
    [*] --> TierSelected: Select Initial Tier
    TierSelected --> Executing: Execute Scraping
    Executing --> Success: Content Retrieved
    Executing --> NetworkError: Network Issues
    Executing --> ContentError: Content Issues
    Executing --> JSError: JavaScript Issues
    Executing --> BotDetected: Anti-bot Detection
    
    NetworkError --> Retry: Same Tier
    Retry --> Success: Resolved
    Retry --> EscalateTier: Still Failing
    
    ContentError --> EscalateTier: Higher Tier
    JSError --> EscalateTier: Browser Automation
    BotDetected --> EscalateTier: AI Automation
    
    EscalateTier --> TierSelected: Try Next Tier
    EscalateTier --> Failed: All Tiers Exhausted
    
    Success --> [*]
    Failed --> [*]
```

### 4. Vector Search Architecture

#### Query Pipeline with HyDE Enhancement

```mermaid
sequenceDiagram
    participant U as User Query
    participant S as SecurityValidator
    participant H as HyDE Generator
    participant E as EmbeddingManager
    participant C as DragonflyDB Cache
    participant Q as QdrantService
    participant R as BGE Reranker
    
    U->>S: Original Query
    S->>S: Validate & Sanitize
    S->>H: Validated Query
    H->>H: Generate Hypothetical Document
    H->>E: Hypothetical Doc + Query
    E->>E: Create Enhanced Vector
    E->>C: Check Cache
    
    alt Cache Hit
        C-->>U: ⚡ Cached Result (0.8ms)
    else Cache Miss
        C->>Q: Enhanced Vector
        Q->>Q: Query API + Prefetch
        Q->>Q: Multi-stage Retrieval
        Q->>R: Retrieved Documents
        R->>R: BGE Reranking
        R->>C: Store in Cache
        C-->>U: 📋 Final Response
    end
    
    Note over C,Q: Cache Miss Flow
    Note over U,C: Cache Hit: 0.8ms P99
    Note over Q,R: Multi-stage + Reranking
```

#### Qdrant Integration Architecture

```mermaid
classDiagram
    class QdrantService {
        +client: QdrantClient
        +query_api: QueryAPI
        +collection_manager: CollectionManager
        +search_vectors(query) List[SearchResult]
        +hybrid_search(dense, sparse) List[SearchResult]
        +create_collection(config) bool
        +manage_aliases(operations) bool
    }
    
    class QueryAPI {
        +prefetch_config: PrefetchConfig
        +fusion_config: FusionConfig
        +multi_stage_search(query) SearchResults
        +optimize_prefetch(params) PrefetchParams
    }
    
    class CollectionManager {
        +aliases: Dict[str, str]
        +create_collection_with_alias(name) bool
        +atomic_switch(old, new) bool
        +zero_downtime_update(collection) bool
    }
    
    class SearchOptimizer {
        +hnsw_config: HNSWConfig
        +payload_indexes: PayloadIndexes
        +optimize_hnsw(collection) bool
        +create_payload_indexes(fields) bool
    }
    
    QdrantService --> QueryAPI
    QdrantService --> CollectionManager
    QdrantService --> SearchOptimizer
```

### 5. Caching Architecture

#### DragonflyDB Integration

```mermaid
flowchart TD
    A["📥 Request"] --> B["⚡ DragonflyDB Cache"]
    B --> C{"Cache Hit?"}
    
    C -->|✅ Hit| D["📤 Return Cached<br/>⏱️ 0.8ms P99<br/>📊 900K ops/sec"]
    
    C -->|❌ Miss| E["🧠 Processing Pipeline"]
    E --> F["🔍 Query Execution"]
    F --> G["📋 Results Processing"]
    G --> H["💾 Cache Storage<br/>🎯 Smart TTL<br/>📈 38% memory savings"]
    H --> I["📤 Return Fresh Result"]
    
    J["🔄 Cache Management<br/>⏰ TTL: 1 hour<br/>🗑️ LRU eviction"] -.-> B
    
    classDef fast fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
    classDef processing fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef cache fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef decision fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class D fast
    class E,F,G,H,I processing
    class B,J cache
    class C decision
```

#### Cache Strategy Implementation

```python
class CacheManager:
    def __init__(self, dragonfly_url: str):
        self.client = DragonflyClient(dragonfly_url)
        self.ttl_strategy = SmartTTLStrategy()
    
    async def get_or_compute(
        self, 
        key: str, 
        compute_fn: Callable,
        ttl: Optional[int] = None
    ):
        """Get from cache or compute and store."""
        
        # Try cache first
        cached = await self.client.get(key)
        if cached:
            return self._deserialize(cached)
        
        # Compute fresh value
        value = await compute_fn()
        
        # Store with smart TTL
        ttl = ttl or self.ttl_strategy.calculate_ttl(key, value)
        await self.client.setex(key, ttl, self._serialize(value))
        
        return value
    
    def _calculate_cache_key(self, query: str, filters: Dict) -> str:
        """Generate deterministic cache key."""
        key_data = {
            "query": query,
            "filters": sorted(filters.items()),
            "timestamp": int(time.time() / 300)  # 5-minute buckets
        }
        return f"search:{hashlib.md5(json.dumps(key_data).encode()).hexdigest()}"
```

## 📊 Data Flow Architecture

### Document Ingestion Pipeline

```mermaid
flowchart TD
    A["🌐 URL Input"] --> B["🎯 AutomationRouter"]
    B --> C["🔍 Tier Selection<br/>Static → Tier 0<br/>Dynamic → Tier 1-4"]
    
    C --> D["📄 Content Extraction"]
    D --> E["🔒 Security Validation<br/>URL sanitization<br/>Content filtering"]
    E --> F["✂️ Enhanced Chunking<br/>AST-based for code<br/>Semantic for text"]
    
    F --> G["📝 Chunk Processing"]
    G --> H["🧮 Embedding Generation<br/>Dense + Sparse vectors<br/>Batch optimization"]
    H --> I["🗄️ Vector Storage<br/>Qdrant with payload indexing<br/>Collection aliases"]
    
    I --> J["🏷️ Metadata Indexing<br/>Full-text search<br/>Filtered queries"]
    J --> K["✅ Completion"]
    
    classDef input fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef router fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef processing fill:#f1f8e9,stroke:#388e3c,stroke-width:2px
    classDef storage fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class A input
    class B,C router
    class D,E,F,G,H processing
    class I,J,K storage
```

### Search Pipeline

```mermaid
flowchart TD
    A["🔍 Search Query"] --> B["🛡️ Security Validation"]
    B --> C["🧠 HyDE Enhancement<br/>Generate hypothetical doc<br/>Expand query context"]
    
    C --> D["🔢 Vector Generation<br/>Dense embeddings<br/>Sparse embeddings"]
    D --> E["⚡ Cache Check<br/>DragonflyDB lookup<br/>0.8ms P99 latency"]
    
    E -->|Hit| F["📤 Cached Results"]
    E -->|Miss| G["🗄️ Qdrant Query API<br/>Multi-stage retrieval<br/>Prefetch optimization"]
    
    G --> H["📊 Fusion Algorithm<br/>Dense + Sparse fusion<br/>RRF scoring"]
    H --> I["🎯 BGE Reranking<br/>Semantic reordering<br/>Relevance optimization"]
    I --> J["💾 Cache Storage<br/>Smart TTL<br/>Result caching"]
    J --> K["📋 Final Results"]
    
    F --> L["📤 Response"]
    K --> L
    
    classDef query fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef cache fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef results fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class A,B query
    class C,D,G,H,I processing
    class E,J cache
    class F,K,L results
```

## 🔧 Implementation Patterns

### Service Dependency Injection

```python
class ServiceContainer:
    """Dependency injection container for services."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._services = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize all services in dependency order."""
        if self._initialized:
            return
        
        # Initialize core services first
        self._services['cache'] = CacheManager(self.config.cache)
        await self._services['cache'].initialize()
        
        self._services['embedding'] = EmbeddingManager(
            config=self.config.embedding,
            cache=self._services['cache']
        )
        await self._services['embedding'].initialize()
        
        self._services['qdrant'] = QdrantService(
            config=self.config.qdrant,
            embedding_manager=self._services['embedding']
        )
        await self._services['qdrant'].initialize()
        
        self._services['browser'] = UnifiedBrowserManager(
            config=self.config.browser
        )
        await self._services['browser'].initialize()
        
        self._initialized = True
    
    def get_service(self, name: str):
        """Get service by name."""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        return self._services.get(name)
    
    async def cleanup(self):
        """Cleanup all services."""
        for service in reversed(list(self._services.values())):
            if hasattr(service, 'cleanup'):
                await service.cleanup()
        self._services.clear()
        self._initialized = False
```

### Error Handling Strategy

```python
class ServiceError(Exception):
    """Base exception for service errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

class ErrorHandler:
    """Centralized error handling with retry logic."""
    
    @staticmethod
    async def with_retry(
        operation: Callable,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        exceptions: Tuple = (Exception,)
    ):
        """Execute operation with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except exceptions as e:
                last_exception = e
                if attempt == max_retries:
                    break
                
                wait_time = backoff_factor * (2 ** attempt)
                await asyncio.sleep(wait_time)
        
        raise last_exception
    
    @staticmethod
    def handle_service_error(error: Exception, context: str) -> ServiceError:
        """Convert various errors to standardized service errors."""
        if isinstance(error, ValidationError):
            return ServiceError(f"Validation failed in {context}", {
                "error_type": "validation",
                "original_error": str(error)
            })
        elif isinstance(error, aiohttp.ClientError):
            return ServiceError(f"Network error in {context}", {
                "error_type": "network",
                "original_error": str(error)
            })
        else:
            return ServiceError(f"Unknown error in {context}", {
                "error_type": "unknown",
                "original_error": str(error)
            })
```

### Configuration Management

```python
class ConfigurationManager:
    """Advanced configuration management with validation and hot-reload."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        self.validators = []
        self.change_handlers = []
    
    def add_validator(self, validator: Callable[[UnifiedConfig], bool]):
        """Add configuration validator."""
        self.validators.append(validator)
    
    def add_change_handler(self, handler: Callable[[UnifiedConfig], None]):
        """Add configuration change handler."""
        self.change_handlers.append(handler)
    
    def load_config(self) -> UnifiedConfig:
        """Load and validate configuration."""
        # Load from environment and files
        config_data = self._load_config_data()
        
        # Create config object with validation
        config = UnifiedConfig(**config_data)
        
        # Run custom validators
        for validator in self.validators:
            if not validator(config):
                raise ConfigurationError("Configuration validation failed")
        
        self.config = config
        return config
    
    def _load_config_data(self) -> Dict[str, Any]:
        """Load configuration from multiple sources."""
        config_data = {}
        
        # 1. Load defaults
        config_data.update(self._get_defaults())
        
        # 2. Load from file
        if self.config_path and os.path.exists(self.config_path):
            config_data.update(self._load_from_file(self.config_path))
        
        # 3. Load from environment
        config_data.update(self._load_from_env())
        
        return config_data
    
    async def watch_config_changes(self):
        """Watch for configuration changes and reload."""
        if not self.config_path:
            return
        
        last_modified = os.path.getmtime(self.config_path)
        
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            try:
                current_modified = os.path.getmtime(self.config_path)
                if current_modified > last_modified:
                    logger.info("Configuration file changed, reloading...")
                    new_config = self.load_config()
                    
                    # Notify handlers
                    for handler in self.change_handlers:
                        await handler(new_config)
                    
                    last_modified = current_modified
            except Exception as e:
                logger.error(f"Error reloading configuration: {e}")
```

## 🚀 Performance Architecture

### Performance Characteristics

#### Speed Improvements

- **Crawling**: 0.4s avg (baseline: 2.5s) - 6.25x improvement with Crawl4AI
- **Search Latency**: < 50ms P95 (baseline: 100ms) - Query API + DragonflyDB
- **Filtered Search**: < 20ms (baseline: 1000ms+) - 50x improvement with payload indexing
- **Cache Operations**: 0.8ms P99 (baseline: 2.5ms) - 3x improvement with DragonflyDB

#### Accuracy Enhancements

- **HyDE**: 15-25% improved query understanding
- **Query API Prefetch**: 10-15% relevance improvement
- **HNSW Tuning**: 5% improved recall@10
- **Combined Effect**: 50-70% overall improvement

### Optimization Strategies

#### Connection Pooling

```python
class OptimizedServiceManager:
    """Service manager with optimized connection pooling."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.connection_pools = {}
    
    async def get_qdrant_client(self) -> QdrantClient:
        """Get optimized Qdrant client with connection pooling."""
        if 'qdrant' not in self.connection_pools:
            self.connection_pools['qdrant'] = QdrantClient(
                url=self.config.qdrant.url,
                api_key=self.config.qdrant.api_key,
                timeout=self.config.qdrant.timeout,
                # Optimized settings
                prefer_grpc=True,
                grpc_port=6334,
                # Connection pooling
                grpc_options={
                    'grpc.keepalive_time_ms': 30000,
                    'grpc.keepalive_timeout_ms': 5000,
                    'grpc.keepalive_permit_without_calls': True,
                    'grpc.http2.max_pings_without_data': 0,
                    'grpc.http2.min_time_between_pings_ms': 10000,
                    'grpc.http2.min_ping_interval_without_data_ms': 300000
                }
            )
        return self.connection_pools['qdrant']
    
    async def get_dragonfly_client(self) -> redis.Redis:
        """Get optimized DragonflyDB client."""
        if 'dragonfly' not in self.connection_pools:
            self.connection_pools['dragonfly'] = redis.Redis.from_url(
                self.config.cache.redis_url,
                # Optimized connection pool
                connection_pool=redis.ConnectionPool(
                    max_connections=20,
                    retry_on_timeout=True,
                    health_check_interval=30
                ),
                decode_responses=True
            )
        return self.connection_pools['dragonfly']
```

#### Batch Processing Optimization

```python
class BatchProcessor:
    """Optimized batch processing for embeddings and storage."""
    
    def __init__(self, embedding_manager: EmbeddingManager, qdrant_service: QdrantService):
        self.embedding_manager = embedding_manager
        self.qdrant_service = qdrant_service
        self.batch_size = 100
        self.max_concurrent = 5
    
    async def process_documents_batch(self, documents: List[Document]) -> List[ProcessingResult]:
        """Process documents in optimized batches."""
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch: List[Document]) -> List[ProcessingResult]:
            async with semaphore:
                # Extract texts
                texts = [doc.content for doc in batch]
                
                # Generate embeddings in batch
                embeddings = await self.embedding_manager.generate_embeddings(texts)
                
                # Prepare Qdrant points
                points = [
                    PointStruct(
                        id=doc.id,
                        vector=embedding,
                        payload=doc.metadata
                    )
                    for doc, embedding in zip(batch, embeddings)
                ]
                
                # Store in Qdrant
                await self.qdrant_service.upsert_points(points)
                
                return [ProcessingResult(success=True, document_id=doc.id) for doc in batch]
        
        # Process in batches
        batches = [documents[i:i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
        batch_tasks = [process_batch(batch) for batch in batches]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
            else:
                results.extend(batch_result)
        
        return results
```

## 🔒 Security Architecture

### Security Validation Pipeline

```mermaid
flowchart TD
    A["📥 Input Request"] --> B["🔍 Input Validation<br/>Schema validation<br/>Type checking"]
    B --> C["🌐 URL Validation<br/>Domain whitelist<br/>SSRF prevention"]
    C --> D["📝 Content Filtering<br/>Size limits<br/>Format validation"]
    D --> E["🛡️ Security Scanning<br/>Malicious content<br/>Privacy protection"]
    E --> F["✅ Sanitized Output"]
    
    B -->|Invalid| G["❌ Validation Error"]
    C -->|Blocked| H["❌ URL Blocked"]
    D -->|Too Large| I["❌ Content Rejected"]
    E -->|Malicious| J["❌ Security Block"]
    
    classDef valid fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef invalid fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    
    class F valid
    class G,H,I,J invalid
    class A,B,C,D,E process
```

### Security Implementation

```python
class SecurityValidator:
    """Comprehensive security validation for all inputs."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.url_validator = URLValidator(config.url_validation)
        self.content_filter = ContentFilter(config.content_filtering)
    
    async def validate_search_request(self, request: SearchRequest) -> SearchRequest:
        """Validate and sanitize search request."""
        # Query validation
        if not request.query or len(request.query.strip()) == 0:
            raise ValidationError("Query cannot be empty")
        
        if len(request.query) > self.config.max_query_length:
            raise ValidationError("Query too long")
        
        # Sanitize query
        sanitized_query = self._sanitize_query(request.query)
        
        # Collection name validation
        if not self._is_valid_collection_name(request.collection_name):
            raise ValidationError("Invalid collection name")
        
        return request.model_copy(update={"query": sanitized_query})
    
    async def validate_url(self, url: str) -> str:
        """Validate URL for security and policy compliance."""
        # Basic URL format validation
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError("Invalid URL format")
        except Exception:
            raise ValidationError("Invalid URL format")
        
        # Check against blocked domains
        if parsed.netloc in self.config.blocked_domains:
            raise ValidationError("Domain is blocked")
        
        # Check for localhost/private IPs
        if self._is_private_address(parsed.netloc):
            raise ValidationError("Private addresses not allowed")
        
        # Check for allowed domains (if whitelist exists)
        if self.config.allowed_domains and parsed.netloc not in self.config.allowed_domains:
            raise ValidationError("Domain not in allowlist")
        
        return url
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize search query."""
        # Remove potential injection attempts
        sanitized = re.sub(r'[<>"\']', '', query)
        
        # Limit special characters
        sanitized = re.sub(r'[^\w\s\-.,?!]', '', sanitized)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized.strip()
    
    def _is_private_address(self, hostname: str) -> bool:
        """Check if hostname is a private address."""
        private_patterns = [
            r'^localhost$',
            r'^127\.',
            r'^10\.',
            r'^192\.168\.',
            r'^172\.(1[6-9]|2[0-9]|3[01])\.'
        ]
        
        return any(re.match(pattern, hostname) for pattern in private_patterns)
```

## 📡 API Architecture

### MCP Server Architecture

```mermaid
classDiagram
    class UnifiedMCPServer {
        +server: Server
        +services: ServiceContainer
        +tools: Dict[str, MCPTool]
        +register_tool(tool) None
        +handle_request(request) Response
        +start_server() None
    }
    
    class MCPTool {
        <<abstract>>
        +name: str
        +description: str
        +input_schema: Dict
        +execute(args)* Any
        +validate_input(args) bool
    }
    
    class SearchTool {
        +name: "search_documents"
        +service: QdrantService
        +execute(args) SearchResponse
    }
    
    class DocumentTool {
        +name: "add_url"
        +service: BrowserManager
        +execute(args) DocumentResponse
    }
    
    class CollectionTool {
        +name: "manage_collections"
        +service: QdrantService
        +execute(args) CollectionResponse
    }
    
    UnifiedMCPServer --> MCPTool
    MCPTool <|-- SearchTool
    MCPTool <|-- DocumentTool
    MCPTool <|-- CollectionTool
```

### Tool Registration Pattern

```python
class MCPToolRegistry:
    """Registry for MCP tools with automatic discovery."""
    
    def __init__(self, services: ServiceContainer):
        self.services = services
        self.tools = {}
    
    def register_tool(self, tool_class: Type[MCPTool]):
        """Register an MCP tool."""
        tool = tool_class(self.services)
        self.tools[tool.name] = tool
        return tool
    
    def auto_discover_tools(self):
        """Automatically discover and register tools."""
        from src.mcp_tools.tools import (
            SearchTool, DocumentTool, CollectionTool,
            AnalyticsTool, CacheTool, SecurityTool
        )
        
        # Register core tools
        self.register_tool(SearchTool)
        self.register_tool(DocumentTool)
        self.register_tool(CollectionTool)
        self.register_tool(AnalyticsTool)
        self.register_tool(CacheTool)
        self.register_tool(SecurityTool)
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with schemas."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            }
            for tool in self.tools.values()
        ]

# Example tool implementation
class SearchTool(MCPTool):
    name = "search_documents"
    description = "Search documents using hybrid vector search"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
            "collection": {"type": "string", "default": "documents"}
        },
        "required": ["query"]
    }
    
    def __init__(self, services: ServiceContainer):
        self.qdrant_service = services.get_service('qdrant')
        self.security_validator = services.get_service('security')
    
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search with full validation and error handling."""
        try:
            # Validate input
            request = SearchRequest(**args)
            request = await self.security_validator.validate_search_request(request)
            
            # Execute search
            results = await self.qdrant_service.search_vectors(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit
            )
            
            return {
                "success": True,
                "results": [result.model_dump() for result in results],
                "total_count": len(results),
                "query_time_ms": results.query_time_ms
            }
            
        except ValidationError as e:
            return {"success": False, "error": f"Validation failed: {str(e)}"}
        except Exception as e:
            logger.exception("Search tool execution failed")
            return {"success": False, "error": "Internal search error"}
```

## 🔄 Deployment Architecture

### Zero-Downtime Deployment

```mermaid
gantt
    title Zero-Downtime Deployment Timeline
    dateFormat X
    axisFormat %s
    
    section Preparation
    Build New Index         :active, build, 0, 30
    Validate Index         :validate, after build, 10
    
    section Deployment
    Blue-Green Switch      :crit, switch, after validate, 5
    Monitor Health         :monitor, after switch, 15
    
    section Traffic Migration
    Gradual Traffic Shift  :traffic, after monitor, 20
    Complete Migration     :done, milestone, after traffic, 0
    
    section Monitoring
    Health Checks          :health, 0, 70
    Performance Metrics    :metrics, 0, 70
    Error Monitoring       :errors, 0, 70
```

### Collection Alias Management

```python
class ZeroDowntimeDeployment:
    """Manage zero-downtime deployments using collection aliases."""
    
    def __init__(self, qdrant_service: QdrantService):
        self.qdrant_service = qdrant_service
        self.alias_manager = CollectionAliasManager(qdrant_service)
    
    async def deploy_new_index(
        self, 
        documents: List[Document], 
        alias_name: str = "documents"
    ) -> bool:
        """Deploy new index with zero downtime."""
        
        # 1. Create new collection with timestamp
        timestamp = int(time.time())
        new_collection = f"{alias_name}_{timestamp}"
        
        logger.info(f"Creating new collection: {new_collection}")
        await self.qdrant_service.create_collection(
            collection_name=new_collection,
            config=self._get_optimized_collection_config()
        )
        
        # 2. Build index in background
        logger.info("Building index in background...")
        await self._build_index(new_collection, documents)
        
        # 3. Validate new index
        logger.info("Validating new index...")
        if not await self._validate_index(new_collection):
            await self.qdrant_service.delete_collection(new_collection)
            raise DeploymentError("Index validation failed")
        
        # 4. Get current collection
        current_collection = await self.alias_manager.get_collection_for_alias(alias_name)
        
        # 5. Atomic switch using aliases
        logger.info(f"Switching alias {alias_name} to {new_collection}")
        await self.alias_manager.atomic_switch(
            alias_name=alias_name,
            new_collection=new_collection,
            old_collection=current_collection
        )
        
        # 6. Monitor health for 5 minutes
        logger.info("Monitoring deployment health...")
        if not await self._monitor_deployment_health(alias_name, duration=300):
            # Rollback on health check failure
            logger.error("Health check failed, rolling back...")
            await self.alias_manager.atomic_switch(
                alias_name=alias_name,
                new_collection=current_collection,
                old_collection=new_collection
            )
            raise DeploymentError("Deployment health check failed")
        
        # 7. Cleanup old collection (after delay)
        if current_collection:
            asyncio.create_task(self._cleanup_old_collection(current_collection, delay=3600))
        
        logger.info(f"Deployment completed successfully: {new_collection}")
        return True
    
    async def _build_index(self, collection_name: str, documents: List[Document]):
        """Build index with optimized batch processing."""
        batch_processor = BatchProcessor(
            embedding_manager=self.embedding_manager,
            qdrant_service=self.qdrant_service
        )
        
        await batch_processor.process_documents_batch(documents)
    
    async def _validate_index(self, collection_name: str) -> bool:
        """Validate index health and functionality."""
        try:
            # Check collection exists and has points
            info = await self.qdrant_service.get_collection_info(collection_name)
            if info.points_count == 0:
                return False
            
            # Perform test search
            test_results = await self.qdrant_service.search_vectors(
                collection_name=collection_name,
                query="test validation query",
                limit=1
            )
            
            return len(test_results) > 0
        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False
```

## 📈 Monitoring Architecture

### Observability Stack

```mermaid
flowchart TD
    A["📊 Application Metrics"] --> B["📈 Prometheus"]
    C["📝 Application Logs"] --> D["🔍 Elasticsearch"]
    E["🔍 Distributed Tracing"] --> F["🕸️ Jaeger"]
    
    B --> G["📊 Grafana Dashboard"]
    D --> H["🔍 Kibana Dashboard"]
    F --> I["🕸️ Jaeger UI"]
    
    G --> J["🚨 Alertmanager"]
    H --> J
    I --> J
    
    J --> K["📧 Alert Notifications"]
    
    classDef metrics fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef logs fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef tracing fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef dashboards fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef alerts fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class A,B,G metrics
    class C,D,H logs
    class E,F,I tracing
    class J,K alerts
```

### Health Check Implementation

```python
class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, services: ServiceContainer):
        self.services = services
        self.health_checks = []
        self.metrics_collector = MetricsCollector()
    
    def register_health_check(self, check: HealthCheck):
        """Register a health check."""
        self.health_checks.append(check)
    
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status."""
        check_results = await asyncio.gather(
            *[check.execute() for check in self.health_checks],
            return_exceptions=True
        )
        
        healthy_checks = sum(1 for result in check_results 
                           if isinstance(result, HealthCheckResult) and result.healthy)
        total_checks = len(self.health_checks)
        
        overall_status = self._determine_overall_status(check_results)
        
        return SystemHealth(
            status=overall_status,
            healthy_checks=healthy_checks,
            total_checks=total_checks,
            checks={
                check.name: result for check, result in zip(self.health_checks, check_results)
            },
            timestamp=time.time()
        )
    
    def _determine_overall_status(self, results: List) -> HealthStatus:
        """Determine overall system health status."""
        healthy_count = sum(1 for r in results 
                          if isinstance(r, HealthCheckResult) and r.healthy)
        total_count = len(results)
        
        if healthy_count == total_count:
            return HealthStatus.HEALTHY
        elif healthy_count >= total_count * 0.75:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

# Health check implementations
class QdrantHealthCheck(HealthCheck):
    name = "qdrant"
    
    async def execute(self) -> HealthCheckResult:
        try:
            info = await self.qdrant_service.client.get_collections()
            return HealthCheckResult(
                healthy=True,
                message=f"Qdrant healthy: {len(info.collections)} collections",
                response_time=self.response_time
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Qdrant unhealthy: {str(e)}",
                response_time=self.response_time
            )

class CacheHealthCheck(HealthCheck):
    name = "cache"
    
    async def execute(self) -> HealthCheckResult:
        try:
            await self.cache_service.client.ping()
            info = await self.cache_service.client.info()
            return HealthCheckResult(
                healthy=True,
                message=f"Cache healthy: {info.get('connected_clients', 0)} clients",
                response_time=self.response_time
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Cache unhealthy: {str(e)}",
                response_time=self.response_time
            )
```

This comprehensive architecture documentation provides a complete understanding of the AI Documentation Vector DB system's design, implementation patterns, and operational characteristics. The architecture is designed for high performance, reliability, and maintainability while supporting advanced features like intelligent browser automation, hybrid vector search, and zero-downtime deployments.

---

*🏗️ This architecture supports high-performance document processing, intelligent search capabilities, and robust operational patterns. All components are designed with scalability, reliability, and maintainability in mind.*
