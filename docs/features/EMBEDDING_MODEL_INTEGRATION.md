# Embedding Model Integration & Optimization Guide

## Overview

This guide details the integration of multiple embedding models, smart model selection strategies, and optimization techniques for our AI Documentation Vector DB. We support OpenAI text-embedding-3 series, BGE models, local FastEmbed models, and provisions for future models like NV-Embed-v2.

## Embedding Model Landscape (2025)

### Performance Comparison

| Model | MTEB Score | Dimensions | Cost/1M tokens | Speed | Use Case |
|-------|------------|------------|----------------|-------|----------|
| text-embedding-3-large | 64.6% | 3072 | $0.13 | Medium | High accuracy |
| text-embedding-3-small | 62.3% | 768 | $0.02 | Fast | General purpose |
| NV-Embed-v2 | 69.3% | 4096 | Varies | Slow | Research/specialized |
| bge-base-en-v1.5 | 63.5% | 768 | Free | Fast | Local deployment |
| bge-large-en-v1.5 | 64.2% | 1024 | Free | Medium | Local high accuracy |
| bge-m3 | 66.0% | 1024 | Free | Medium | Multilingual |

## Architecture Design

### Embedding Service Architecture

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np

class EmbeddingModel(Enum):
    """Available embedding models."""
    # OpenAI Models
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    
    # BGE Models
    BGE_BASE_EN = "BAAI/bge-base-en-v1.5"
    BGE_LARGE_EN = "BAAI/bge-large-en-v1.5"
    BGE_M3 = "BAAI/bge-m3"
    BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
    
    # FastEmbed Models
    FASTEMBED_BGE_BASE = "fastembed-bge-base-en-v1.5"
    
    # Future Models
    NV_EMBED_V2 = "nvidia/nv-embed-v2"

class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed_texts(
        self,
        texts: List[str],
        model: str,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    @abstractmethod
    async def estimate_cost(
        self,
        texts: List[str],
        model: str
    ) -> float:
        """Estimate cost for embedding generation."""
        pass
```

## Provider Implementations

### 1. OpenAI Provider

```python
from openai import AsyncOpenAI
import tiktoken
import asyncio
from typing import List, Optional

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with advanced features."""
    
    def __init__(self, api_key: str, organization: Optional[str] = None):
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.model_info = {
            "text-embedding-3-large": {
                "dimensions": 3072,
                "max_tokens": 8191,
                "cost_per_million": 0.13,
                "supports_dimensions": True,
                "min_dimensions": 256
            },
            "text-embedding-3-small": {
                "dimensions": 768,
                "max_tokens": 8191,
                "cost_per_million": 0.02,
                "supports_dimensions": True,
                "min_dimensions": 256
            }
        }
    
    async def embed_texts(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int] = None,
        batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate embeddings with batching and dimension control.
        
        Args:
            texts: List of texts to embed
            model: Model name
            dimensions: Optional dimensions (for Matryoshka)
            batch_size: Batch size for API calls
        """
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Truncate texts if needed
            truncated_batch = [
                self._truncate_text(text, model) 
                for text in batch
            ]
            
            # API call with retry logic
            response = await self._embed_with_retry(
                truncated_batch,
                model,
                dimensions
            )
            
            embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
    
    async def _embed_with_retry(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int],
        max_retries: int = 3
    ):
        """Embed with exponential backoff retry."""
        
        for attempt in range(max_retries):
            try:
                kwargs = {"model": model, "input": texts}
                if dimensions:
                    kwargs["dimensions"] = dimensions
                
                return await self.client.embeddings.create(**kwargs)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                    
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
    
    def _truncate_text(self, text: str, model: str) -> str:
        """Truncate text to model's token limit."""
        
        max_tokens = self.model_info[model]["max_tokens"]
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) > max_tokens:
            # Keep some tokens for safety margin
            tokens = tokens[:max_tokens - 10]
            text = self.tokenizer.decode(tokens)
        
        return text
    
    async def estimate_cost(
        self,
        texts: List[str],
        model: str
    ) -> float:
        """Estimate embedding generation cost."""
        
        total_tokens = 0
        for text in texts:
            tokens = len(self.tokenizer.encode(text))
            total_tokens += tokens
        
        cost_per_million = self.model_info[model]["cost_per_million"]
        estimated_cost = (total_tokens / 1_000_000) * cost_per_million
        
        return estimated_cost
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model information."""
        return self.model_info.get(model, {})
```

### 2. Local Model Provider (BGE/FastEmbed)

```python
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
import torch
from typing import List, Optional
import numpy as np

class LocalEmbeddingProvider(EmbeddingProvider):
    """Local model provider for BGE and FastEmbed models."""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = cache_dir
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_info = {
            "BAAI/bge-base-en-v1.5": {
                "dimensions": 768,
                "max_tokens": 512,
                "cost_per_million": 0,
                "type": "sentence-transformers"
            },
            "BAAI/bge-large-en-v1.5": {
                "dimensions": 1024,
                "max_tokens": 512,
                "cost_per_million": 0,
                "type": "sentence-transformers"
            },
            "BAAI/bge-m3": {
                "dimensions": 1024,
                "max_tokens": 8192,
                "cost_per_million": 0,
                "type": "sentence-transformers",
                "multilingual": True
            },
            "fastembed-bge-base-en-v1.5": {
                "dimensions": 768,
                "max_tokens": 512,
                "cost_per_million": 0,
                "type": "fastembed"
            }
        }
    
    async def embed_texts(
        self,
        texts: List[str],
        model: str,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings using local models."""
        
        # Load model if not cached
        if model not in self.models:
            self._load_model(model)
        
        model_type = self.model_info[model]["type"]
        
        if model_type == "sentence-transformers":
            return await self._embed_sentence_transformers(
                texts, model, normalize
            )
        elif model_type == "fastembed":
            return await self._embed_fastembed(
                texts, model
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_model(self, model_name: str):
        """Load model into memory."""
        
        model_type = self.model_info[model_name]["type"]
        
        if model_type == "sentence-transformers":
            self.models[model_name] = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
        elif model_type == "fastembed":
            self.models[model_name] = TextEmbedding(
                model_name=model_name.replace("fastembed-", ""),
                cache_dir=self.cache_dir
            )
    
    async def _embed_sentence_transformers(
        self,
        texts: List[str],
        model: str,
        normalize: bool
    ) -> np.ndarray:
        """Embed using sentence-transformers."""
        
        model_obj = self.models[model]
        
        # Encode with proper parameters
        embeddings = model_obj.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )
        
        return embeddings
    
    async def _embed_fastembed(
        self,
        texts: List[str],
        model: str
    ) -> np.ndarray:
        """Embed using FastEmbed."""
        
        model_obj = self.models[model]
        
        # FastEmbed returns generator, convert to numpy
        embeddings = list(model_obj.embed(texts))
        return np.array(embeddings)
    
    async def estimate_cost(
        self,
        texts: List[str],
        model: str
    ) -> float:
        """Local models have no API cost."""
        return 0.0
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model information."""
        return self.model_info.get(model, {})
```

### 3. Smart Model Selection

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics

@dataclass
class ModelSelectionCriteria:
    """Criteria for model selection."""
    accuracy_weight: float = 0.4
    speed_weight: float = 0.3
    cost_weight: float = 0.3
    max_cost_per_million: Optional[float] = None
    require_multilingual: bool = False
    require_local: bool = False
    min_accuracy: float = 0.6

class SmartModelSelector:
    """Intelligent model selection based on requirements."""
    
    def __init__(self, providers: Dict[str, EmbeddingProvider]):
        self.providers = providers
        self.model_benchmarks = self._load_benchmarks()
        
    def select_model(
        self,
        texts: List[str],
        criteria: ModelSelectionCriteria = None
    ) -> str:
        """
        Select optimal model based on texts and criteria.
        
        Returns:
            Model name
        """
        
        if criteria is None:
            criteria = ModelSelectionCriteria()
        
        # Analyze text characteristics
        text_stats = self._analyze_texts(texts)
        
        # Get candidate models
        candidates = self._get_candidate_models(text_stats, criteria)
        
        # Score each model
        scores = {}
        for model in candidates:
            score = self._score_model(model, text_stats, criteria)
            scores[model] = score
        
        # Select best model
        best_model = max(scores, key=scores.get)
        
        return best_model
    
    def _analyze_texts(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text characteristics."""
        
        lengths = [len(text) for text in texts]
        
        # Detect language
        languages = []
        for text in texts[:10]:  # Sample first 10
            lang = self._detect_language(text)
            languages.append(lang)
        
        primary_language = max(set(languages), key=languages.count)
        is_multilingual = len(set(languages)) > 1
        
        return {
            "count": len(texts),
            "avg_length": statistics.mean(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths),
            "primary_language": primary_language,
            "is_multilingual": is_multilingual,
            "contains_code": any(self._contains_code(t) for t in texts[:10])
        }
    
    def _get_candidate_models(
        self,
        text_stats: Dict[str, Any],
        criteria: ModelSelectionCriteria
    ) -> List[str]:
        """Get candidate models based on requirements."""
        
        candidates = []
        
        # Filter by requirements
        for model, info in self.model_benchmarks.items():
            # Check multilingual requirement
            if criteria.require_multilingual and not info.get("multilingual"):
                continue
            
            # Check local requirement
            if criteria.require_local and info.get("provider") != "local":
                continue
            
            # Check cost constraint
            if criteria.max_cost_per_million:
                if info["cost_per_million"] > criteria.max_cost_per_million:
                    continue
            
            # Check accuracy threshold
            if info["accuracy"] < criteria.min_accuracy:
                continue
            
            # Check token limits
            if text_stats["max_length"] > info["max_chars"]:
                continue
            
            candidates.append(model)
        
        return candidates
    
    def _score_model(
        self,
        model: str,
        text_stats: Dict[str, Any],
        criteria: ModelSelectionCriteria
    ) -> float:
        """Score model based on criteria."""
        
        info = self.model_benchmarks[model]
        
        # Normalize scores to 0-1
        accuracy_score = info["accuracy"]
        speed_score = 1 / (1 + info["latency_ms"] / 100)  # Lower is better
        cost_score = 1 / (1 + info["cost_per_million"])   # Lower is better
        
        # Apply weights
        total_score = (
            criteria.accuracy_weight * accuracy_score +
            criteria.speed_weight * speed_score +
            criteria.cost_weight * cost_score
        )
        
        # Bonus for matching characteristics
        if text_stats["is_multilingual"] and info.get("multilingual"):
            total_score *= 1.2
        
        if text_stats["contains_code"] and "code" in info.get("strengths", []):
            total_score *= 1.1
        
        return total_score
    
    def _load_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Load model benchmark data."""
        
        return {
            "text-embedding-3-large": {
                "accuracy": 0.646,
                "latency_ms": 150,
                "cost_per_million": 0.13,
                "max_chars": 32000,
                "provider": "openai",
                "strengths": ["accuracy", "context"]
            },
            "text-embedding-3-small": {
                "accuracy": 0.623,
                "latency_ms": 50,
                "cost_per_million": 0.02,
                "max_chars": 32000,
                "provider": "openai",
                "strengths": ["speed", "cost"]
            },
            "BAAI/bge-base-en-v1.5": {
                "accuracy": 0.635,
                "latency_ms": 30,
                "cost_per_million": 0,
                "max_chars": 2000,
                "provider": "local",
                "strengths": ["speed", "privacy"]
            },
            "BAAI/bge-m3": {
                "accuracy": 0.660,
                "latency_ms": 80,
                "cost_per_million": 0,
                "max_chars": 8000,
                "provider": "local",
                "multilingual": True,
                "strengths": ["multilingual", "accuracy"]
            }
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # In production, use langdetect or similar
        import re
        
        # Check for CJK characters
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text):
            return "multilingual"
        
        # Check for non-ASCII
        if not text.isascii():
            return "multilingual"
        
        return "en"
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code."""
        
        code_indicators = [
            "def ", "class ", "import ", "function",
            "{", "}", "=>", "```"
        ]
        
        return any(indicator in text for indicator in code_indicators)
```

## Optimization Strategies

### 1. Batch Processing Optimization

```python
class BatchedEmbeddingService:
    """Optimized batch processing for embeddings."""
    
    def __init__(self, providers: Dict[str, EmbeddingProvider]):
        self.providers = providers
        self.batch_queue = asyncio.Queue()
        self.processing = False
        
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str,
        priority: int = 0
    ) -> np.ndarray:
        """Generate embeddings with intelligent batching."""
        
        # For high priority, process immediately
        if priority > 5 or len(texts) > 100:
            return await self._process_immediate(texts, model)
        
        # Otherwise, add to batch queue
        future = asyncio.Future()
        await self.batch_queue.put({
            "texts": texts,
            "model": model,
            "future": future,
            "priority": priority
        })
        
        # Start batch processor if not running
        if not self.processing:
            asyncio.create_task(self._batch_processor())
        
        return await future
    
    async def _batch_processor(self):
        """Process batches efficiently."""
        
        self.processing = True
        batch_timeout = 0.1  # 100ms
        max_batch_size = 1000
        
        while True:
            batch = []
            start_time = asyncio.get_event_loop().time()
            
            # Collect items for batch
            while len(batch) < max_batch_size:
                try:
                    timeout = batch_timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    
                    if timeout <= 0:
                        break
                    
                    item = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=timeout
                    )
                    batch.append(item)
                    
                except asyncio.TimeoutError:
                    break
            
            if not batch:
                self.processing = False
                break
            
            # Group by model
            model_groups = {}
            for item in batch:
                model = item["model"]
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append(item)
            
            # Process each model group
            for model, items in model_groups.items():
                all_texts = []
                indices = []
                
                for i, item in enumerate(items):
                    start_idx = len(all_texts)
                    all_texts.extend(item["texts"])
                    end_idx = len(all_texts)
                    indices.append((start_idx, end_idx))
                
                # Generate embeddings for all texts
                try:
                    provider = self._get_provider(model)
                    all_embeddings = await provider.embed_texts(
                        all_texts, model
                    )
                    
                    # Distribute results
                    for i, (start, end) in enumerate(indices):
                        embeddings = all_embeddings[start:end]
                        items[i]["future"].set_result(embeddings)
                        
                except Exception as e:
                    # Set exception for all items
                    for item in items:
                        item["future"].set_exception(e)
```

### 2. Caching Strategy

```python
import hashlib
import pickle
from datetime import datetime, timedelta

class EmbeddingCache:
    """Intelligent embedding cache with TTL and compression."""
    
    def __init__(self, redis_client, compression: bool = True):
        self.redis = redis_client
        self.compression = compression
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: str
    ) -> Optional[np.ndarray]:
        """Retrieve cached embeddings."""
        
        cache_keys = [
            self._generate_cache_key(text, model)
            for text in texts
        ]
        
        # Batch get from Redis
        cached_values = await self.redis.mget(cache_keys)
        
        # Check if all are cached
        embeddings = []
        for i, cached in enumerate(cached_values):
            if cached is None:
                self.stats["misses"] += 1
                return None  # Cache miss
            
            embedding = self._deserialize(cached)
            embeddings.append(embedding)
        
        self.stats["hits"] += len(embeddings)
        return np.array(embeddings)
    
    async def cache_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        model: str,
        ttl: int = 86400  # 24 hours default
    ):
        """Cache embeddings with TTL."""
        
        # Prepare batch operation
        pipeline = self.redis.pipeline()
        
        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text, model)
            serialized = self._serialize(embeddings[i])
            
            pipeline.setex(cache_key, ttl, serialized)
        
        await pipeline.execute()
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate deterministic cache key."""
        
        # Normalize text
        normalized = text.strip().lower()
        
        # Create hash
        content = f"{model}:{normalized}"
        hash_key = hashlib.sha256(content.encode()).hexdigest()
        
        return f"emb:v1:{model}:{hash_key[:16]}"
    
    def _serialize(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding with optional compression."""
        
        if self.compression:
            import zlib
            data = pickle.dumps(embedding)
            return zlib.compress(data)
        else:
            return pickle.dumps(embedding)
    
    def _deserialize(self, data: bytes) -> np.ndarray:
        """Deserialize embedding."""
        
        if self.compression:
            import zlib
            data = zlib.decompress(data)
        
        return pickle.loads(data)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests if total_requests > 0 else 0
        )
        
        return {
            "hit_rate": hit_rate,
            "total_hits": self.stats["hits"],
            "total_misses": self.stats["misses"],
            "total_evictions": self.stats["evictions"]
        }
```

### 3. Cost Optimization

```python
class CostOptimizedEmbeddingService:
    """Service with cost optimization strategies."""
    
    def __init__(
        self,
        providers: Dict[str, EmbeddingProvider],
        budget_tracker: BudgetTracker
    ):
        self.providers = providers
        self.budget_tracker = budget_tracker
        self.model_selector = SmartModelSelector(providers)
        
    async def generate_embeddings_with_budget(
        self,
        texts: List[str],
        max_budget: float,
        min_quality: float = 0.6
    ) -> np.ndarray:
        """Generate embeddings within budget constraints."""
        
        # Estimate costs for different models
        cost_estimates = {}
        for model in self.get_available_models():
            provider = self._get_provider(model)
            cost = await provider.estimate_cost(texts, model)
            cost_estimates[model] = cost
        
        # Filter by budget
        affordable_models = {
            model: cost for model, cost in cost_estimates.items()
            if cost <= max_budget
        }
        
        if not affordable_models:
            raise ValueError(
                f"No models available within budget ${max_budget}"
            )
        
        # Select best model within budget
        criteria = ModelSelectionCriteria(
            min_accuracy=min_quality,
            cost_weight=0.5,  # Higher weight on cost
            accuracy_weight=0.3,
            speed_weight=0.2
        )
        
        selected_model = self.model_selector.select_model(
            texts, criteria
        )
        
        # Check if selected model is within budget
        if selected_model not in affordable_models:
            # Fallback to cheapest affordable model
            selected_model = min(
                affordable_models,
                key=affordable_models.get
            )
        
        # Track budget usage
        actual_cost = cost_estimates[selected_model]
        await self.budget_tracker.track_usage(
            model=selected_model,
            cost=actual_cost,
            tokens=sum(len(t) for t in texts)
        )
        
        # Generate embeddings
        provider = self._get_provider(selected_model)
        embeddings = await provider.embed_texts(texts, selected_model)
        
        return embeddings
```

### 4. Quality Assurance

```python
class EmbeddingQualityMonitor:
    """Monitor and ensure embedding quality."""
    
    def __init__(self):
        self.test_queries = self._load_test_queries()
        self.baseline_scores = {}
        
    async def validate_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        model: str
    ) -> Dict[str, Any]:
        """Validate embedding quality."""
        
        validations = {
            "dimension_check": self._check_dimensions(embeddings, model),
            "norm_check": self._check_norms(embeddings),
            "diversity_check": self._check_diversity(embeddings),
            "semantic_check": await self._check_semantic_quality(
                embeddings, texts, model
            )
        }
        
        overall_valid = all(v["valid"] for v in validations.values())
        
        return {
            "valid": overall_valid,
            "validations": validations,
            "model": model,
            "sample_size": len(embeddings)
        }
    
    def _check_dimensions(
        self,
        embeddings: np.ndarray,
        model: str
    ) -> Dict[str, Any]:
        """Check embedding dimensions."""
        
        expected_dims = self._get_expected_dimensions(model)
        actual_dims = embeddings.shape[1]
        
        return {
            "valid": actual_dims == expected_dims,
            "expected": expected_dims,
            "actual": actual_dims
        }
    
    def _check_norms(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Check embedding norms."""
        
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Most embeddings should be normalized
        normalized = np.allclose(norms, 1.0, atol=0.01)
        
        return {
            "valid": normalized or np.all(norms > 0),
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "normalized": normalized
        }
    
    def _check_diversity(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Check embedding diversity."""
        
        if len(embeddings) < 2:
            return {"valid": True, "message": "Too few embeddings"}
        
        # Calculate pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)
        
        # Remove diagonal
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        off_diagonal = similarities[mask]
        
        # Check if embeddings are too similar
        mean_similarity = float(np.mean(off_diagonal))
        max_similarity = float(np.max(off_diagonal))
        
        # Embeddings shouldn't be identical
        valid = max_similarity < 0.999
        
        return {
            "valid": valid,
            "mean_similarity": mean_similarity,
            "max_similarity": max_similarity,
            "warning": "Embeddings too similar" if not valid else None
        }
    
    async def _check_semantic_quality(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        model: str
    ) -> Dict[str, Any]:
        """Check semantic quality using test queries."""
        
        if model not in self.baseline_scores:
            return {
                "valid": True,
                "message": "No baseline for comparison"
            }
        
        # Run semantic similarity tests
        test_scores = []
        
        for test_case in self.test_queries[:5]:  # Sample tests
            query_text = test_case["query"]
            relevant_text = test_case["relevant"]
            
            # Find most similar text
            query_embedding = await self._get_single_embedding(
                query_text, model
            )
            
            similarities = np.dot(embeddings, query_embedding)
            most_similar_idx = np.argmax(similarities)
            most_similar_text = texts[most_similar_idx]
            
            # Check if it matches expected
            is_correct = most_similar_text == relevant_text
            test_scores.append(is_correct)
        
        accuracy = sum(test_scores) / len(test_scores) if test_scores else 0
        baseline = self.baseline_scores[model]
        
        return {
            "valid": accuracy >= baseline * 0.9,  # 90% of baseline
            "accuracy": accuracy,
            "baseline": baseline,
            "tests_run": len(test_scores)
        }
```

## Monitoring and Analytics

```python
class EmbeddingServiceMetrics:
    """Comprehensive metrics for embedding service."""
    
    def __init__(self, prometheus_client):
        self.prom = prometheus_client
        
        # Define metrics
        self.embedding_latency = self.prom.Histogram(
            'embedding_generation_latency_seconds',
            'Embedding generation latency',
            ['model', 'provider'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.embedding_batch_size = self.prom.Histogram(
            'embedding_batch_size',
            'Batch size for embedding generation',
            ['model'],
            buckets=[1, 10, 50, 100, 500, 1000]
        )
        
        self.model_usage = self.prom.Counter(
            'embedding_model_usage_total',
            'Total usage by model',
            ['model', 'provider']
        )
        
        self.cost_tracker = self.prom.Counter(
            'embedding_cost_dollars',
            'Cumulative cost in dollars',
            ['model']
        )
        
        self.cache_metrics = self.prom.Counter(
            'embedding_cache_operations_total',
            'Cache operations',
            ['operation', 'status']
        )
    
    async def record_generation(
        self,
        model: str,
        provider: str,
        batch_size: int,
        latency: float,
        cost: float
    ):
        """Record embedding generation metrics."""
        
        self.embedding_latency.labels(
            model=model,
            provider=provider
        ).observe(latency)
        
        self.embedding_batch_size.labels(
            model=model
        ).observe(batch_size)
        
        self.model_usage.labels(
            model=model,
            provider=provider
        ).inc(batch_size)
        
        self.cost_tracker.labels(
            model=model
        ).inc(cost)
```

## Best Practices

### 1. Model Selection Guidelines

- **High Accuracy Required**: Use text-embedding-3-large or NV-Embed-v2
- **Cost Sensitive**: Use text-embedding-3-small or BGE models
- **Low Latency**: Use local BGE models with GPU
- **Multilingual**: Use BGE-M3 or text-embedding-3 models
- **Privacy Concerns**: Use local models exclusively

### 2. Optimization Tips

- **Batch Requests**: Always batch embedding generation (optimal: 50-100 texts)
- **Cache Aggressively**: Cache embeddings with appropriate TTL
- **Use Matryoshka**: Leverage dimension reduction for large-scale search
- **Monitor Costs**: Track usage and set budget alerts
- **Validate Quality**: Regularly test embedding quality

### 3. Error Handling

- **Implement Retries**: Use exponential backoff for API failures
- **Fallback Models**: Have backup models for critical operations
- **Graceful Degradation**: Continue with partial results when possible
- **Monitor Failures**: Track and alert on error rates

### 4. Performance Targets

- **Latency**: < 100ms for small batches (< 10 texts)
- **Throughput**: > 1000 embeddings/second with batching
- **Cache Hit Rate**: > 80% for repeated queries
- **Cost Efficiency**: < $50 per million embeddings average

## Configurable Model Benchmarks

### Overview

The embedding system now supports configurable model benchmarks, allowing users to customize model performance data without code changes. This enables:

- **Custom Model Support**: Add new models with performance characteristics
- **Benchmark Updates**: Update model performance data as new benchmarks become available
- **Environment-Specific Tuning**: Adjust model selection based on deployment environment
- **A/B Testing**: Compare different benchmark configurations

### Configuration Structure

Model benchmarks are defined in the `EmbeddingConfig.model_benchmarks` field:

```json
{
  "embedding": {
    "model_benchmarks": {
      "text-embedding-3-small": {
        "model_name": "text-embedding-3-small",
        "provider": "openai",
        "avg_latency_ms": 78,
        "quality_score": 85,
        "tokens_per_second": 12800,
        "cost_per_million_tokens": 20.0,
        "max_context_length": 8191,
        "embedding_dimensions": 1536
      },
      "custom-model": {
        "model_name": "custom-model",
        "provider": "fastembed",
        "avg_latency_ms": 45,
        "quality_score": 82,
        "tokens_per_second": 25000,
        "cost_per_million_tokens": 0.0,
        "max_context_length": 1024,
        "embedding_dimensions": 768
      }
    }
  }
}
```

### Field Descriptions

- **model_name**: Unique identifier for the model
- **provider**: Provider name (openai, fastembed, custom)
- **avg_latency_ms**: Average response time in milliseconds
- **quality_score**: Quality rating 0-100 based on retrieval accuracy
- **tokens_per_second**: Processing throughput
- **cost_per_million_tokens**: Cost per million tokens (0 for local models)
- **max_context_length**: Maximum input context in tokens
- **embedding_dimensions**: Vector dimensionality

### Default Benchmarks

The system provides research-backed default benchmarks for:

- **text-embedding-3-small**: OpenAI's cost-effective model
- **text-embedding-3-large**: OpenAI's high-accuracy model  
- **BAAI/bge-small-en-v1.5**: FastEmbed local small model
- **BAAI/bge-large-en-v1.5**: FastEmbed local large model

### Customization Examples

#### Adding a New Model

```python
from src.config.models import ModelBenchmark, UnifiedConfig

config = UnifiedConfig()
config.embedding.model_benchmarks["my-model"] = ModelBenchmark(
    model_name="my-model",
    provider="custom",
    avg_latency_ms=60,
    quality_score=88,
    tokens_per_second=18000,
    cost_per_million_tokens=5.0,
    max_context_length=2048,
    embedding_dimensions=1024
)
```

#### Updating Performance Data

```python
# Update latency based on new benchmarks
config.embedding.model_benchmarks["text-embedding-3-small"].avg_latency_ms = 65
config.embedding.model_benchmarks["text-embedding-3-small"].quality_score = 87
```

#### Environment-Specific Configuration

Create different config files for different environments:

**config/production-benchmarks.json**:
```json
{
  "embedding": {
    "model_benchmarks": {
      "text-embedding-3-small": {
        "avg_latency_ms": 95,
        "quality_score": 85
      }
    }
  }
}
```

**config/development-benchmarks.json**:
```json
{
  "embedding": {
    "model_benchmarks": {
      "text-embedding-3-small": {
        "avg_latency_ms": 60,
        "quality_score": 85
      }
    }
  }
}
```

### Smart Selection Impact

Configurable benchmarks directly affect smart model selection:

- **Quality Priority**: Models with higher `quality_score` preferred for quality tiers
- **Speed Priority**: Models with lower `avg_latency_ms` preferred for speed
- **Cost Optimization**: Models with lower `cost_per_million_tokens` preferred for budget constraints
- **Capability Matching**: Models with appropriate `max_context_length` selected for text size

### Validation

The system validates benchmark configurations:

```python
# All fields are required and validated
ModelBenchmark(
    model_name="test",           # Required string
    provider="test",             # Required string  
    avg_latency_ms=100,          # Must be > 0
    quality_score=85,            # Must be 0-100
    tokens_per_second=1000,      # Must be > 0
    cost_per_million_tokens=10,  # Must be >= 0
    max_context_length=512,      # Must be > 0
    embedding_dimensions=768     # Must be > 0
)
```

### Usage in Practice

1. **Create custom configuration template** from `config/templates/custom-benchmarks.json`
2. **Update benchmark values** based on your environment's performance
3. **Load configuration** using `UnifiedConfig.load_from_file()`
4. **Test smart selection** to verify model choices meet expectations

This flexible benchmark system ensures optimal model selection across different deployment scenarios while maintaining performance and cost efficiency.

---

This comprehensive guide provides a robust foundation for embedding model integration with flexibility for future enhancements and model additions.
