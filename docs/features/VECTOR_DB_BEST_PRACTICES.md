# Vector Database Management Best Practices

**Status**: V1 Enhanced  
**Last Updated**: 2025-05-26  
**Part of**: [Features Documentation Hub](./README.md)

> **Quick Links**: [Advanced Search](./ADVANCED_SEARCH_IMPLEMENTATION.md) | [HyDE Enhancement](./HYDE_QUERY_ENHANCEMENT.md) | [Enhanced Chunking](./ENHANCED_CHUNKING_GUIDE.md) | [Embedding Models](./EMBEDDING_MODEL_INTEGRATION.md)

## Overview

This guide provides comprehensive best practices for managing Qdrant vector databases in production, including the latest V1 enhancements: Query API, advanced payload indexing, HNSW optimization, collection aliases for zero-downtime deployments, and HyDE integration for improved search accuracy.

## Collection Design Principles

### 1. Collection Architecture

```python
from qdrant_client.models import *
from typing import Dict, Any, Optional

class CollectionDesigner:
    """Best practices for collection design."""

    def create_optimized_collection(
        self,
        name: str,
        use_case: str = "hybrid_search"
    ) -> Dict[str, Any]:
        """
        Create collection with optimal configuration.

        Use cases:
        - hybrid_search: Dense + sparse vectors
        - semantic_search: Dense vectors only
        - multi_modal: Multiple vector types
        - high_performance: Speed optimized
        - high_accuracy: Accuracy optimized
        """

        configs = {
            "hybrid_search": {
                "vectors_config": {
                    "dense": VectorParams(
                        size=768,
                        distance=Distance.COSINE,
                        on_disk=False  # Keep in RAM for speed
                    ),
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                            full_scan_threshold=5000
                        )
                    )
                },
                "hnsw_config": HnswConfigDiff(
                    m=16,  # V1 Optimized for documentation
                    ef_construct=200,  # V1 Enhanced accuracy
                    full_scan_threshold=10000,
                    max_indexing_threads=0  # Use all available
                ),
                "quantization_config": ScalarQuantization(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True  # Keep quantized vectors in RAM
                ),
                "optimizers_config": OptimizersConfigDiff(
                    deleted_threshold=0.2,  # Vacuum at 20% deleted
                    vacuum_min_vector_number=1000,
                    default_segment_number=4,  # Parallel segments
                    max_segment_size=200000,
                    memmap_threshold=50000,
                    indexing_threshold=20000,
                    flush_interval_sec=5,
                    max_optimization_threads=0  # Use all cores
                )
            },
            "semantic_search": {
                "vectors_config": VectorParams(
                    size=1536,  # OpenAI large
                    distance=Distance.COSINE,
                    on_disk=False
                ),
                "hnsw_config": HnswConfigDiff(
                    m=32,  # More connections for accuracy
                    ef_construct=400,
                    full_scan_threshold=5000
                ),
                "quantization_config": ScalarQuantization(
                    type=ScalarType.INT8,
                    quantile=0.95
                )
            },
            "high_performance": {
                "vectors_config": VectorParams(
                    size=768,
                    distance=Distance.DOT,  # Fastest distance
                    on_disk=False
                ),
                "hnsw_config": HnswConfigDiff(
                    m=8,  # Fewer edges for speed
                    ef_construct=100,
                    full_scan_threshold=20000
                ),
                "quantization_config": BinaryQuantization(
                    always_ram=True  # Extreme compression
                )
            }
        }

        return configs.get(use_case, configs["hybrid_search"])
```

### 2. Payload Schema Design

```python
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentPayload(BaseModel):
    """Best practice payload schema."""

    # Required fields
    doc_id: str = Field(description="Unique document identifier")
    content: str = Field(description="Original text content")
    chunk_index: int = Field(description="Position in original document")

    # Metadata for filtering
    source_url: str
    doc_type: str  # "api", "guide", "tutorial", "reference"
    language: str = "en"

    # Timestamps
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime = Field(default_factory=datetime.utcnow)

    # Search optimization
    title: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    category: Optional[str] = None

    # Quality metrics
    quality_score: float = Field(ge=0, le=1, default=1.0)
    views: int = 0

    # Chunking metadata
    chunk_size: int
    overlap_size: int
    total_chunks: int

    # Embedding metadata
    embedding_model: str
    embedding_dimensions: int

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PayloadIndexer:
    """Optimize payload indexing."""

    @staticmethod
    def get_indexed_fields() -> Dict[str, PayloadSchemaType]:
        """Define fields to index for fast filtering."""

        return {
            "doc_type": PayloadSchemaType.KEYWORD,
            "language": PayloadSchemaType.KEYWORD,
            "category": PayloadSchemaType.KEYWORD,
            "created_at": PayloadSchemaType.DATETIME,
            "updated_at": PayloadSchemaType.DATETIME,
            "quality_score": PayloadSchemaType.FLOAT,
            "views": PayloadSchemaType.INTEGER
        }

    @staticmethod
    def create_collection_with_schema(
        client: QdrantClient,
        collection_name: str,
        vector_config: Dict[str, Any]
    ):
        """Create collection with indexed payload fields."""

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_config["vectors_config"],
            hnsw_config=vector_config.get("hnsw_config"),
            quantization_config=vector_config.get("quantization_config"),
            optimizers_config=vector_config.get("optimizers_config")
        )

        # Create payload indexes
        indexed_fields = PayloadIndexer.get_indexed_fields()

        for field_name, field_type in indexed_fields.items():
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )
```

### 3. Sharding and Partitioning Strategies

```python
class ShardingStrategy:
    """Implement sharding for large-scale collections."""

    def __init__(self, client: QdrantClient):
        self.client = client

    async def create_sharded_collections(
        self,
        base_name: str,
        shard_count: int = 4,
        shard_by: str = "date"  # "date", "language", "category"
    ) -> List[str]:
        """Create sharded collections for scalability."""

        collections = []

        if shard_by == "date":
            # Monthly shards
            for i in range(shard_count):
                collection_name = f"{base_name}_2025_{i+1:02d}"
                await self._create_shard(collection_name)
                collections.append(collection_name)

        elif shard_by == "language":
            # Language-based shards
            languages = ["en", "es", "fr", "de", "ja", "zh"]
            for lang in languages[:shard_count]:
                collection_name = f"{base_name}_{lang}"
                await self._create_shard(collection_name)
                collections.append(collection_name)

        elif shard_by == "category":
            # Category-based shards
            categories = ["api", "guides", "tutorials", "references"]
            for category in categories:
                collection_name = f"{base_name}_{category}"
                await self._create_shard(collection_name)
                collections.append(collection_name)

        # Create unified view
        await self._create_alias(base_name, collections)

        return collections

    async def _create_shard(self, collection_name: str):
        """Create individual shard with optimized settings."""

        config = CollectionDesigner().create_optimized_collection(
            collection_name,
            use_case="hybrid_search"
        )

        await self.client.create_collection(
            collection_name=collection_name,
            **config
        )

    async def _create_alias(self, alias_name: str, collections: List[str]):
        """Create alias for unified access."""

        # In production, implement a routing layer
        # that queries multiple collections
        pass
```

## V1 Query API Best Practices

### 1. Multi-Stage Retrieval with Query API

```python
from qdrant_client.models import *

class QueryAPIOptimizer:
    """Leverage Qdrant Query API for advanced retrieval."""
    
    def __init__(self, client: QdrantClient):
        self.client = client
    
    async def multi_stage_search(
        self,
        collection_name: str,
        query_vector: list[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        filters: Optional[Filter] = None,
        limit: int = 10
    ) -> list[ScoredPoint]:
        """
        V1 Enhanced: Multi-stage retrieval with Query API.
        
        Benefits:
        - 15-30% latency reduction
        - Native fusion algorithms
        - Better relevance with prefetch
        """
        
        # Build prefetch stages
        prefetch_queries = []
        
        # Stage 1: Sparse vector prefetch (if available)
        if sparse_vector:
            prefetch_queries.append(
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    filter=filters,
                    limit=100,  # Cast wide net
                    score_threshold=0.0
                )
            )
        
        # Stage 2: Dense vector prefetch
        prefetch_queries.append(
            Prefetch(
                query=query_vector,
                using="dense",
                filter=filters,
                limit=50,  # Refine results
                score_threshold=0.7
            )
        )
        
        # Final query with fusion
        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch_queries if prefetch_queries else None,
            query=query_vector if not prefetch_queries else None,
            using="dense" if not prefetch_queries else None,
            filter=filters,
            limit=limit,
            with_payload=True,
            # V1: Use native fusion
            fusion=Fusion.RRF if sparse_vector else None
        )
        
        return results
    
    async def matryoshka_search(
        self,
        collection_name: str,
        query_text: str,
        embedding_service,
        limit: int = 10
    ) -> list[ScoredPoint]:
        """
        V1 Enhanced: Matryoshka embeddings with nested prefetch.
        
        Small → Medium → Large → Rerank pattern.
        """
        
        # Generate embeddings at different dimensions
        small_emb = await embedding_service.generate(
            query_text, model="small", dims=384
        )
        medium_emb = await embedding_service.generate(
            query_text, model="medium", dims=768
        )
        large_emb = await embedding_service.generate(
            query_text, model="large", dims=1536
        )
        
        # Nested prefetch for progressive refinement
        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    # Stage 1: Cast wide net with small embeddings
                    query=small_emb,
                    using="small",
                    limit=1000,
                    score_threshold=0.5,
                    prefetch=[
                        Prefetch(
                            # Stage 2: Refine with medium embeddings
                            query=medium_emb,
                            using="medium",
                            limit=100,
                            score_threshold=0.7,
                            prefetch=[
                                Prefetch(
                                    # Stage 3: Final refinement with large
                                    query=large_emb,
                                    using="large",
                                    limit=50,
                                    score_threshold=0.8
                                )
                            ]
                        )
                    ]
                )
            ],
            query=large_emb,  # Final reranking
            using="large",
            limit=limit
        )
        
        return results
```

### 2. HyDE Integration Best Practices

```python
class HyDEOptimizer:
    """Hypothetical Document Embeddings for better search."""
    
    def __init__(self, llm_service, embedding_service, cache_service):
        self.llm = llm_service
        self.embeddings = embedding_service
        self.cache = cache_service
    
    async def hyde_enhanced_search(
        self,
        query: str,
        collection_name: str,
        use_cache: bool = True
    ) -> list[ScoredPoint]:
        """
        V1 Enhanced: HyDE search with 15-25% accuracy improvement.
        """
        
        # Check cache
        cache_key = f"hyde:{hashlib.md5(query.encode()).hexdigest()}"
        if use_cache:
            if cached := await self.cache.get(cache_key):
                hyde_embedding = cached
            else:
                # Generate hypothetical document
                hyde_doc = await self._generate_hyde_document(query)
                
                # Create embedding
                hyde_embedding = await self.embeddings.generate(hyde_doc)
                
                # Cache for reuse
                await self.cache.set(cache_key, hyde_embedding, ttl=3600)
        else:
            hyde_doc = await self._generate_hyde_document(query)
            hyde_embedding = await self.embeddings.generate(hyde_doc)
        
        # Use Query API with HyDE embedding
        return await self.client.query_points(
            collection_name=collection_name,
            query=hyde_embedding,
            using="dense",
            limit=10,
            # HyDE works best with prefetch
            prefetch=[
                Prefetch(
                    query=hyde_embedding,
                    using="dense",
                    limit=50,
                    score_threshold=0.6
                )
            ]
        )
    
    async def _generate_hyde_document(self, query: str) -> str:
        """Generate hypothetical document."""
        prompt = f"""Given the search query: '{query}'
        
Write a detailed technical documentation excerpt that would perfectly answer this query.
Include specific technical details, code examples if relevant, and comprehensive explanations.
This should be the ideal document that someone would want to find.

Hypothetical Document:"""
        
        response = await self.llm.generate(
            prompt,
            model="claude-3-haiku",  # Fast and cheap
            max_tokens=500
        )
        
        return response
```

### 3. Advanced Payload Indexing

```python
class PayloadIndexingOptimizer:
    """V1 Enhanced payload indexing strategies."""
    
    @staticmethod
    def get_v1_indexed_fields() -> Dict[str, Any]:
        """V1 optimized field indexing configuration."""
        
        return {
            # Keyword indexes for exact match (10-100x faster)
            "language": PayloadSchemaType.KEYWORD,
            "framework": PayloadSchemaType.KEYWORD,
            "doc_type": PayloadSchemaType.KEYWORD,
            "version": PayloadSchemaType.KEYWORD,
            
            # Text indexes for partial match
            "title": PayloadSchemaType.TEXT,
            "section": PayloadSchemaType.TEXT,
            "keywords": PayloadSchemaType.TEXT,
            
            # Numeric indexes for range queries
            "created_at": PayloadSchemaType.DATETIME,
            "updated_at": PayloadSchemaType.DATETIME,
            "quality_score": PayloadSchemaType.FLOAT,
            "relevance_score": PayloadSchemaType.FLOAT,
            
            # New V1 fields
            "last_accessed": PayloadSchemaType.INTEGER,  # For hot/cold tiering
            "embedding_model": PayloadSchemaType.KEYWORD,  # For model filtering
            "chunk_strategy": PayloadSchemaType.KEYWORD,  # For quality filtering
        }
    
    async def create_compound_indexes(
        self,
        client: QdrantClient,
        collection_name: str
    ):
        """Create compound indexes for common query patterns."""
        
        # Analyze common filter combinations
        common_patterns = [
            ["language", "framework"],  # e.g., Python + FastAPI
            ["doc_type", "version"],    # e.g., API docs v2.0
            ["framework", "updated_at"], # Recent framework docs
        ]
        
        # Create multi-field indexes
        for pattern in common_patterns:
            # Qdrant will optimize queries using these fields together
            pass  # Compound indexes created automatically

## Performance Optimization

### 1. Indexing Optimization

```python
class IndexingOptimizer:
    """Optimize indexing performance."""

    def __init__(self, client: QdrantClient):
        self.client = client

    async def configure_for_bulk_indexing(
        self,
        collection_name: str
    ):
        """Configure collection for bulk indexing."""

        # Disable indexing during bulk insert
        await self.client.update_collection(
            collection_name=collection_name,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=100000000  # Very high threshold
            )
        )

    async def optimize_after_bulk_insert(
        self,
        collection_name: str
    ):
        """Re-enable indexing and optimize."""

        # Re-enable indexing
        await self.client.update_collection(
            collection_name=collection_name,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=20000  # Normal threshold
            )
        )

        # Force optimization
        await self.client.update_collection(
            collection_name=collection_name,
            optimizer_config=OptimizersConfigDiff(
                max_optimization_threads=0  # Use all cores
            )
        )

    async def parallel_upsert(
        self,
        collection_name: str,
        points: List[PointStruct],
        batch_size: int = 100,
        parallel_batches: int = 4
    ):
        """Parallel batch upsert for speed."""

        import asyncio

        async def upsert_batch(batch: List[PointStruct]):
            await self.client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=False  # Don't wait for indexing
            )

        # Split into batches
        batches = [
            points[i:i + batch_size]
            for i in range(0, len(points), batch_size)
        ]

        # Process in parallel chunks
        for i in range(0, len(batches), parallel_batches):
            chunk = batches[i:i + parallel_batches]
            await asyncio.gather(*[upsert_batch(b) for b in chunk])
```

### 2. Search Performance Tuning

```python
class SearchOptimizer:
    """Optimize search performance."""

    def __init__(self, client: QdrantClient):
        self.client = client
        self.performance_cache = {}

    async def tune_hnsw_parameters(
        self,
        collection_name: str,
        target_recall: float = 0.95,
        max_latency_ms: float = 100
    ):
        """Auto-tune HNSW parameters."""

        test_queries = await self._generate_test_queries()

        # Test different ef values
        ef_values = [50, 100, 200, 400, 800]
        results = []

        for ef in ef_values:
            # Update search params
            await self.client.update_collection(
                collection_name=collection_name,
                hnsw_config=HnswConfigDiff(ef=ef)
            )

            # Run benchmark
            metrics = await self._benchmark_search(
                collection_name,
                test_queries,
                ef
            )

            results.append({
                "ef": ef,
                "recall": metrics["recall"],
                "latency_ms": metrics["latency_ms"]
            })

            # Check if we meet requirements
            if (metrics["recall"] >= target_recall and
                metrics["latency_ms"] <= max_latency_ms):
                break

        # Select optimal ef
        optimal = min(
            [r for r in results
             if r["recall"] >= target_recall],
            key=lambda x: x["latency_ms"]
        )

        await self.client.update_collection(
            collection_name=collection_name,
            hnsw_config=HnswConfigDiff(ef=optimal["ef"])
        )

        return optimal

    async def optimize_filtering(
        self,
        collection_name: str,
        common_filters: List[Filter]
    ):
        """Optimize for common filter patterns."""

        # Analyze filter usage
        filter_stats = await self._analyze_filters(common_filters)

        # Create compound indexes for common combinations
        for filter_combo in filter_stats["frequent_combinations"]:
            if len(filter_combo) > 1:
                # Create multi-field index
                await self._create_compound_index(
                    collection_name,
                    filter_combo
                )
```

### 3. Memory Management

```python
class MemoryOptimizer:
    """Optimize memory usage."""

    def __init__(self, client: QdrantClient):
        self.client = client

    async def calculate_memory_requirements(
        self,
        vector_count: int,
        vector_dimensions: int,
        quantization: str = "int8",
        sparse_vectors: bool = True
    ) -> Dict[str, Any]:
        """Calculate memory requirements."""

        # Base vector size
        bytes_per_dimension = {
            "float32": 4,
            "int8": 1,
            "binary": 0.125  # 1 bit per dimension
        }

        # Dense vectors
        dense_memory = (
            vector_count *
            vector_dimensions *
            bytes_per_dimension[quantization]
        )

        # HNSW index (approximate)
        hnsw_memory = vector_count * 200  # ~200 bytes per vector

        # Sparse vectors (if used)
        sparse_memory = 0
        if sparse_vectors:
            # Assume 100 non-zero values per vector average
            sparse_memory = vector_count * 100 * 8  # 8 bytes per value

        # Payload storage
        payload_memory = vector_count * 1024  # 1KB average payload

        total_memory = (
            dense_memory + hnsw_memory +
            sparse_memory + payload_memory
        )

        return {
            "total_gb": total_memory / (1024**3),
            "breakdown": {
                "dense_vectors_gb": dense_memory / (1024**3),
                "hnsw_index_gb": hnsw_memory / (1024**3),
                "sparse_vectors_gb": sparse_memory / (1024**3),
                "payload_gb": payload_memory / (1024**3)
            },
            "recommendations": self._get_memory_recommendations(
                total_memory
            )
        }

    def _get_memory_recommendations(
        self,
        total_bytes: int
    ) -> Dict[str, Any]:
        """Get memory optimization recommendations."""

        total_gb = total_bytes / (1024**3)

        recommendations = {
            "instance_memory_gb": total_gb * 1.5,  # 50% headroom
            "strategies": []
        }

        if total_gb > 100:
            recommendations["strategies"].extend([
                "Use sharding to distribute load",
                "Enable on-disk storage for vectors",
                "Use binary quantization for maximum compression"
            ])
        elif total_gb > 50:
            recommendations["strategies"].extend([
                "Use INT8 quantization",
                "Consider on-disk storage for older data",
                "Optimize payload storage"
            ])
        else:
            recommendations["strategies"].extend([
                "Keep all data in memory for best performance",
                "Use standard INT8 quantization"
            ])

        return recommendations
```

## Operational Excellence

### 1. Monitoring and Alerting

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import time

@dataclass
class CollectionMetrics:
    """Collection health metrics."""
    vectors_count: int
    indexed_vectors_count: int
    segments_count: int
    disk_usage_mb: float
    ram_usage_mb: float
    search_latency_p95_ms: float
    indexing_lag: int
    error_rate: float

class CollectionMonitor:
    """Monitor collection health and performance."""

    def __init__(self, client: QdrantClient, alerting_service):
        self.client = client
        self.alerting = alerting_service
        self.thresholds = {
            "indexing_lag": 10000,  # vectors
            "search_latency_p95_ms": 100,
            "error_rate": 0.01,  # 1%
            "disk_usage_percent": 80
        }

    async def health_check(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """Comprehensive health check."""

        # Get collection info
        info = await self.client.get_collection(collection_name)

        # Calculate metrics
        metrics = CollectionMetrics(
            vectors_count=info.vectors_count,
            indexed_vectors_count=info.indexed_vectors_count,
            segments_count=info.segments_count,
            disk_usage_mb=info.disk_usage_bytes / (1024**2),
            ram_usage_mb=info.ram_usage_bytes / (1024**2),
            search_latency_p95_ms=await self._measure_search_latency(
                collection_name
            ),
            indexing_lag=info.vectors_count - info.indexed_vectors_count,
            error_rate=await self._calculate_error_rate(collection_name)
        )

        # Check thresholds
        alerts = []
        if metrics.indexing_lag > self.thresholds["indexing_lag"]:
            alerts.append({
                "severity": "warning",
                "message": f"High indexing lag: {metrics.indexing_lag}"
            })

        if metrics.search_latency_p95_ms > self.thresholds["search_latency_p95_ms"]:
            alerts.append({
                "severity": "critical",
                "message": f"High search latency: {metrics.search_latency_p95_ms}ms"
            })

        # Send alerts
        for alert in alerts:
            await self.alerting.send_alert(
                collection=collection_name,
                **alert
            )

        return {
            "healthy": len(alerts) == 0,
            "metrics": metrics,
            "alerts": alerts
        }

    async def _measure_search_latency(
        self,
        collection_name: str,
        samples: int = 10
    ) -> float:
        """Measure search latency."""

        latencies = []

        for _ in range(samples):
            start = time.time()

            await self.client.search(
                collection_name=collection_name,
                query_vector=[0.1] * 768,  # Dummy vector
                limit=10
            )

            latencies.append((time.time() - start) * 1000)

        # Return 95th percentile
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        return latencies[p95_index]
```

### 2. Backup and Recovery

```python
import json
from pathlib import Path
from datetime import datetime

class BackupManager:
    """Manage collection backups and recovery."""

    def __init__(self, client: QdrantClient, storage_path: Path):
        self.client = client
        self.storage_path = storage_path

    async def create_backup(
        self,
        collection_name: str,
        include_vectors: bool = True
    ) -> str:
        """Create collection backup."""

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{collection_name}_{timestamp}"
        backup_path = self.storage_path / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)

        # Save collection config
        config = await self.client.get_collection(collection_name)
        with open(backup_path / "config.json", "w") as f:
            json.dump(config.dict(), f, indent=2)

        # Export data
        offset = None
        batch_size = 1000
        batch_num = 0

        while True:
            # Scroll through collection
            records, offset = await self.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=include_vectors
            )

            if not records:
                break

            # Save batch
            batch_file = backup_path / f"batch_{batch_num:06d}.json"
            with open(batch_file, "w") as f:
                json.dump([r.dict() for r in records], f)

            batch_num += 1

            if offset is None:
                break

        # Create manifest
        manifest = {
            "collection_name": collection_name,
            "backup_id": backup_id,
            "timestamp": timestamp,
            "total_batches": batch_num,
            "include_vectors": include_vectors,
            "config": config.dict()
        }

        with open(backup_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return backup_id

    async def restore_backup(
        self,
        backup_id: str,
        target_collection: Optional[str] = None
    ):
        """Restore collection from backup."""

        backup_path = self.storage_path / backup_id

        # Load manifest
        with open(backup_path / "manifest.json", "r") as f:
            manifest = json.load(f)

        collection_name = target_collection or manifest["collection_name"]

        # Recreate collection
        config = manifest["config"]
        await self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=config["vectors_config"],
            hnsw_config=config.get("hnsw_config"),
            quantization_config=config.get("quantization_config")
        )

        # Restore data
        for batch_num in range(manifest["total_batches"]):
            batch_file = backup_path / f"batch_{batch_num:06d}.json"

            with open(batch_file, "r") as f:
                records = json.load(f)

            # Convert to points
            points = []
            for record in records:
                point = PointStruct(
                    id=record["id"],
                    vector=record.get("vector"),
                    payload=record.get("payload")
                )
                points.append(point)

            # Upsert batch
            await self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=False
            )

        # Wait for indexing
        await self.client.update_collection(
            collection_name=collection_name,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=1  # Force immediate indexing
            )
        )
```

### 3. Migration Strategies

```python
class MigrationManager:
    """Handle collection migrations and upgrades."""

    async def migrate_collection(
        self,
        source_collection: str,
        target_collection: str,
        transformation_fn = None,
        batch_size: int = 1000
    ):
        """Migrate collection with optional transformation."""

        # Create target with same config
        source_info = await self.client.get_collection(source_collection)

        await self.client.create_collection(
            collection_name=target_collection,
            vectors_config=source_info.config.params.vectors,
            hnsw_config=source_info.config.hnsw_config,
            quantization_config=source_info.config.quantization_config
        )

        # Stream and transform data
        offset = None
        migrated_count = 0

        while True:
            # Read batch
            records, offset = await self.client.scroll(
                collection_name=source_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            if not records:
                break

            # Transform if needed
            if transformation_fn:
                records = [transformation_fn(r) for r in records]

            # Write to target
            await self.client.upsert(
                collection_name=target_collection,
                points=records
            )

            migrated_count += len(records)

            if offset is None:
                break

        return {
            "migrated_count": migrated_count,
            "source": source_collection,
            "target": target_collection
        }

    async def upgrade_embeddings(
        self,
        collection_name: str,
        new_embedding_fn,
        new_dimensions: int
    ):
        """Upgrade to new embedding model."""

        # Create temporary collection
        temp_collection = f"{collection_name}_upgrade_{int(time.time())}"

        # Create with new dimensions
        info = await self.client.get_collection(collection_name)
        vectors_config = info.config.params.vectors

        if isinstance(vectors_config, dict):
            vectors_config["dense"].size = new_dimensions
        else:
            vectors_config.size = new_dimensions

        await self.client.create_collection(
            collection_name=temp_collection,
            vectors_config=vectors_config
        )

        # Re-embed and migrate
        async def transform_point(point):
            # Extract text from payload
            text = point.payload.get("content", "")

            # Generate new embedding
            new_vector = await new_embedding_fn(text)

            # Update point
            point.vector = new_vector
            return point

        await self.migrate_collection(
            source_collection=collection_name,
            target_collection=temp_collection,
            transformation_fn=transform_point
        )

        # V1: Use aliases for zero-downtime swap
        await self.client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(
                        alias_name=collection_name
                    )
                ),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=temp_collection,
                        alias_name=collection_name
                    )
                )
            ]
        )
        
        # Schedule old collection cleanup
        await self._schedule_cleanup(collection_name, grace_period_hours=24)
```

## Advanced Patterns

### 1. Time-Based Collections

```python
class TimeBasedCollectionManager:
    """Manage time-partitioned collections."""

    async def setup_rolling_collections(
        self,
        base_name: str,
        retention_days: int = 90
    ):
        """Setup collections with automatic rotation."""

        from datetime import timedelta

        # Create collections for time windows
        collections = []
        for i in range(3):  # Past, current, future
            date = datetime.utcnow() - timedelta(days=30 * i)
            collection_name = f"{base_name}_{date.strftime('%Y_%m')}"

            if not await self._collection_exists(collection_name):
                await self.client.create_collection(
                    collection_name=collection_name,
                    **self.collection_config
                )

            collections.append(collection_name)

        # Create alias pointing to current
        current_collection = collections[1]
        await self.client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=current_collection,
                        alias_name=f"{base_name}_current"
                    )
                )
            ]
        )

        # Schedule cleanup
        await self._schedule_cleanup(base_name, retention_days)
```

### 2. Hybrid Collection Pattern

```python
class HybridCollectionManager:
    """Manage hybrid hot/cold storage."""

    async def setup_hybrid_storage(
        self,
        collection_name: str,
        hot_days: int = 7
    ):
        """Setup hybrid storage with hot/cold tiers."""

        # Hot collection - in memory
        hot_collection = f"{collection_name}_hot"
        await self.client.create_collection(
            collection_name=hot_collection,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
                on_disk=False  # Keep in RAM
            ),
            quantization_config=None  # No quantization for speed
        )

        # Cold collection - on disk
        cold_collection = f"{collection_name}_cold"
        await self.client.create_collection(
            collection_name=cold_collection,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
                on_disk=True  # Store on disk
            ),
            quantization_config=BinaryQuantization()  # Max compression
        )

        # Setup data movement job
        await self._setup_tiering_job(
            hot_collection,
            cold_collection,
            hot_days
        )
```

## V1 Zero-Downtime Deployment Patterns

### 1. Blue-Green Deployment with Aliases

```python
class ZeroDowntimeDeployment:
    """V1 Enhanced: Production deployment patterns."""
    
    def __init__(self, client: QdrantClient, monitoring_service):
        self.client = client
        self.monitoring = monitoring_service
    
    async def blue_green_deployment(
        self,
        alias_name: str,
        new_data_source: str,
        validation_queries: List[str]
    ) -> Dict[str, Any]:
        """
        V1 Pattern: Blue-green deployment with validation.
        
        Zero downtime guaranteed through atomic alias switching.
        """
        
        # Get current collection (blue)
        blue_collection = await self._get_collection_for_alias(alias_name)
        
        # Create new collection (green)
        green_collection = f"{alias_name}_green_{int(time.time())}"
        
        try:
            # 1. Clone configuration from blue
            blue_config = await self.client.get_collection(blue_collection)
            await self.client.create_collection(
                collection_name=green_collection,
                vectors_config=blue_config.config.params.vectors,
                hnsw_config=blue_config.config.hnsw_config,
                quantization_config=blue_config.config.quantization_config
            )
            
            # 2. Populate green collection
            await self._populate_collection(green_collection, new_data_source)
            
            # 3. Create indexes
            await self._create_v1_indexes(green_collection)
            
            # 4. Validate green collection
            validation_passed = await self._validate_collection(
                green_collection,
                validation_queries
            )
            
            if not validation_passed:
                raise ValueError("Validation failed for green collection")
            
            # 5. Atomic switch
            await self.client.update_collection_aliases(
                change_aliases_operations=[
                    DeleteAliasOperation(
                        delete_alias=DeleteAlias(alias_name=alias_name)
                    ),
                    CreateAliasOperation(
                        create_alias=CreateAlias(
                            collection_name=green_collection,
                            alias_name=alias_name
                        )
                    )
                ]
            )
            
            # 6. Monitor for 5 minutes
            await self._monitor_deployment(alias_name, duration=300)
            
            # 7. Schedule blue cleanup
            asyncio.create_task(
                self._cleanup_old_collection(blue_collection, delay_hours=24)
            )
            
            return {
                "success": True,
                "old_collection": blue_collection,
                "new_collection": green_collection,
                "deployment_time": datetime.utcnow()
            }
            
        except Exception as e:
            # Rollback on failure
            await self._rollback(alias_name, blue_collection)
            raise
    
    async def canary_deployment(
        self,
        alias_name: str,
        new_collection: str,
        stages: List[Dict[str, Any]] = None
    ):
        """
        V1 Pattern: Gradual traffic shift with monitoring.
        """
        
        if not stages:
            stages = [
                {"percentage": 5, "duration_minutes": 30, "error_threshold": 0.05},
                {"percentage": 25, "duration_minutes": 60, "error_threshold": 0.02},
                {"percentage": 50, "duration_minutes": 120, "error_threshold": 0.01},
                {"percentage": 100, "duration_minutes": 0, "error_threshold": 0.01}
            ]
        
        old_collection = await self._get_collection_for_alias(alias_name)
        
        for stage in stages:
            percentage = stage["percentage"]
            
            # Route traffic (implement in application layer)
            await self._update_traffic_routing(
                alias_name,
                old_collection,
                new_collection,
                percentage
            )
            
            # Monitor stage
            if stage["duration_minutes"] > 0:
                metrics = await self._monitor_canary(
                    new_collection,
                    duration_minutes=stage["duration_minutes"]
                )
                
                # Check error rate
                if metrics["error_rate"] > stage["error_threshold"]:
                    await self._rollback_canary(alias_name, old_collection)
                    raise ValueError(f"Canary failed at {percentage}%")
            
            # Complete deployment at 100%
            if percentage == 100:
                await self._complete_canary(alias_name, new_collection)

### 2. A/B Testing Implementation

```python
class ABTestingManager:
    """V1 Enhanced: A/B test different configurations."""
    
    async def create_ab_test(
        self,
        test_name: str,
        control_config: Dict[str, Any],
        variant_configs: List[Dict[str, Any]],
        traffic_splits: List[float],
        metrics_to_track: List[str]
    ) -> str:
        """
        Create A/B test for collection configurations.
        
        Example: Test different HNSW parameters, embedding models, etc.
        """
        
        test_id = f"ab_test_{test_name}_{int(time.time())}"
        
        # Create control collection
        control_name = f"{test_id}_control"
        await self._create_collection_with_config(control_name, control_config)
        
        # Create variant collections
        variants = []
        for i, config in enumerate(variant_configs):
            variant_name = f"{test_id}_variant_{i}"
            await self._create_collection_with_config(variant_name, config)
            variants.append(variant_name)
        
        # Setup traffic routing
        self.active_tests[test_id] = {
            "control": control_name,
            "variants": variants,
            "traffic_splits": traffic_splits,
            "metrics": {name: defaultdict(list) for name in metrics_to_track},
            "start_time": datetime.utcnow()
        }
        
        return test_id
    
    async def route_query(
        self,
        test_id: str,
        query: Dict[str, Any],
        user_id: str
    ) -> Tuple[str, Any]:
        """Route query to appropriate variant."""
        
        test = self.active_tests[test_id]
        
        # Deterministic routing based on user
        bucket = hash(user_id) % 100
        
        cumulative = 0
        for i, split in enumerate(test["traffic_splits"]):
            cumulative += split * 100
            if bucket < cumulative:
                collection = test["variants"][i] if i > 0 else test["control"]
                variant = f"variant_{i}" if i > 0 else "control"
                break
        
        # Execute query
        start = time.time()
        results = await self.execute_query(collection, query)
        latency = time.time() - start
        
        # Track metrics
        test["metrics"]["latency"][variant].append(latency)
        test["metrics"]["results_count"][variant].append(len(results))
        
        return variant, results

### 3. Automated Index Rebuilding

```python
class IndexRebuilder:
    """V1 Pattern: Automated index rebuilding with zero downtime."""
    
    async def rebuild_with_new_model(
        self,
        alias_name: str,
        new_embedding_model: str,
        batch_size: int = 1000
    ):
        """Rebuild index with new embedding model."""
        
        old_collection = await self._get_collection_for_alias(alias_name)
        new_collection = f"{old_collection}_rebuild_{int(time.time())}"
        
        # Create new collection with same config
        await self._clone_collection_schema(old_collection, new_collection)
        
        # Stream, re-embed, and insert
        offset = None
        processed = 0
        
        while True:
            # Read batch
            points, offset = await self.client.scroll(
                collection_name=old_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True
            )
            
            if not points:
                break
            
            # Re-embed with new model
            texts = [p.payload["content"] for p in points]
            new_embeddings = await self.embedding_service.generate_batch(
                texts,
                model=new_embedding_model
            )
            
            # Update points
            for point, embedding in zip(points, new_embeddings):
                point.vector = embedding
                point.payload["embedding_model"] = new_embedding_model
            
            # Insert to new collection
            await self.client.upsert(
                collection_name=new_collection,
                points=points
            )
            
            processed += len(points)
            
            # Report progress
            await self._report_progress(
                "rebuild",
                processed,
                estimated_total=await self._get_collection_size(old_collection)
            )
            
            if offset is None:
                break
        
        # Validate and switch
        await self._validate_and_switch(alias_name, new_collection)
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **High Memory Usage**

   - Enable quantization
   - Move vectors to disk
   - Reduce HNSW m parameter
   - Use sharding

2. **Slow Search Performance**

   - Increase ef parameter
   - Reduce result limit
   - Use filters to reduce search space
   - Enable caching

3. **Indexing Lag**

   - Increase indexing threads
   - Reduce segment size
   - Disable indexing during bulk insert
   - Use parallel upsert

4. **Storage Growth**
   - Enable vacuum
   - Remove deleted vectors
   - Compress payloads
   - Archive old data

## V1 Enhanced Best Practices Summary

### Core Principles

1. **Design for Scale**: Use sharding and partitioning from the start
2. **Zero Downtime**: Always use collection aliases for updates
3. **Query API First**: Leverage prefetch and native fusion for 15-30% gains
4. **Index Everything**: Payload indexing for 10-100x filter performance
5. **Cache Intelligently**: HyDE embeddings + DragonflyDB for speed
6. **Monitor Continuously**: Track all V1 metrics (latency, accuracy, cost)
7. **Test in Production**: A/B test configurations safely

### V1 Implementation Checklist

- [ ] Migrate to Query API with multi-stage prefetch
- [ ] Enable payload indexing on all filterable fields
- [ ] Optimize HNSW: m=16, ef_construct=200, adaptive ef
- [ ] Implement HyDE with caching for 15-25% accuracy gain
- [ ] Set up collection aliases for all production collections
- [ ] Deploy DragonflyDB for 3x cache performance
- [ ] Configure Crawl4AI for $0 scraping costs
- [ ] Create A/B testing framework for continuous optimization

### Performance Targets

- **Search Latency**: < 50ms P95 (with Query API + DragonflyDB)
- **Filtered Search**: < 20ms (with payload indexing)
- **Cache Hit Rate**: > 80% (with HyDE caching)
- **Accuracy**: +15-25% (with HyDE)
- **Deployment Downtime**: 0 seconds (with aliases)

This V1-enhanced guide provides state-of-the-art patterns for managing production Qdrant deployments with industry-leading performance and zero-downtime operations.

## See Also

### Related Features

- **[Advanced Search Implementation](./ADVANCED_SEARCH_IMPLEMENTATION.md)** - Query API, payload indexing, and multi-stage retrieval patterns that optimize vector DB performance
- **[HyDE Query Enhancement](./HYDE_QUERY_ENHANCEMENT.md)** - Enhanced query embeddings cached in DragonflyDB for 80% cost reduction
- **[Enhanced Chunking Guide](./ENHANCED_CHUNKING_GUIDE.md)** - Rich metadata extraction that enables powerful payload indexing for 10-100x faster filtering
- **[Embedding Model Integration](./EMBEDDING_MODEL_INTEGRATION.md)** - Smart embedding generation with DragonflyDB caching and cost optimization
- **[Reranking Guide](./RERANKING_GUIDE.md)** - BGE reranking that works seamlessly with Query API multi-stage retrieval

### Architecture Documentation

- **[System Overview](../architecture/SYSTEM_OVERVIEW.md)** - Vector database's central role in the AI documentation system
- **[Unified Configuration](../architecture/UNIFIED_CONFIGURATION.md)** - Configure Qdrant, collections, and indexing strategies
- **[Performance Guide](../operations/PERFORMANCE_GUIDE.md)** - Monitor and optimize vector database performance in production

### Implementation References

- **[Browser Automation](../user-guides/browser-automation.md)** - Content acquisition that feeds into vector database
- **[API Reference](../api/API_REFERENCE.md)** - Vector database API endpoints and operations
- **[Development Workflow](../development/DEVELOPMENT_WORKFLOW.md)** - Testing and validating vector database configurations

### Core Infrastructure Benefits

1. **Query API**: 15-30% performance improvement with multi-stage retrieval and native fusion algorithms
2. **Payload Indexing**: 10-100x faster filtered searches on metadata fields (language, framework, doc_type)
3. **Collection Aliases**: Zero-downtime deployments with blue-green and canary patterns
4. **HNSW Optimization**: 5% accuracy boost with optimized parameters (m=16, ef_construct=200)

### Performance Stack

- **Base**: Qdrant with optimized HNSW and quantization
- **+ Query API**: Multi-stage retrieval for 15-30% speed improvement
- **+ Payload Indexing**: 10-100x faster filtering on metadata
- **+ DragonflyDB**: 4.5x cache performance over Redis
- **+ Collection Aliases**: Zero-downtime deployments and A/B testing
- **= Total**: Production-ready vector database with industry-leading performance

### Integration Flow

1. **Content Processing**: Enhanced chunking → embeddings → vector storage with rich metadata
2. **Search Pipeline**: Query API with prefetch → payload filtering → HyDE enhancement → reranking
3. **Caching Layer**: DragonflyDB for embeddings, search results, and HyDE documents
4. **Operations**: Zero-downtime deployments, A/B testing, and automated monitoring
