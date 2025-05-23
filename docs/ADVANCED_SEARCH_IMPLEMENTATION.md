# Advanced Search & Reranking Implementation Guide

## Overview

This guide details the implementation of advanced search capabilities using Qdrant's Query API, hybrid search techniques, and multi-stage reranking. Our implementation leverages the latest features including RRF/DBSF fusion, sparse vectors, and BGE-reranker-v2-m3.

## Search Architecture

### Search Pipeline Flow

```
Query Input
    ↓
Query Analysis & Enhancement
    ↓
Embedding Generation (Dense + Sparse)
    ↓
Multi-Stage Retrieval
    ├── Stage 1: Broad Retrieval (1000 results)
    ├── Stage 2: Refined Search (100 results)
    └── Stage 3: Reranking (10 results)
    ↓
Result Formatting & Caching
    ↓
Response
```

## Core Search Implementations

### 1. Hybrid Search with Qdrant Query API

```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import *
import numpy as np
from typing import List, Dict, Any

class HybridSearchService:
    def __init__(self, qdrant_client: AsyncQdrantClient):
        self.client = qdrant_client
        self.sparse_encoder = SPLADEEncoder()  # or BM25
        self.dense_encoder = DenseEncoder()
        
    async def hybrid_search(
        self,
        query: str,
        collection: str,
        fusion_method: str = "rrf",
        prefetch_limit: int = 100,
        final_limit: int = 10,
        filters: Filter = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Qdrant Query API with prefetch.
        
        Args:
            query: Search query text
            collection: Collection name
            fusion_method: "rrf" or "dbsf"
            prefetch_limit: Number of candidates per vector type
            final_limit: Final number of results
            filters: Optional Qdrant filters
        """
        
        # Generate embeddings
        dense_vector = await self.dense_encoder.encode(query)
        sparse_vector = await self.sparse_encoder.encode(query)
        
        # Build query request
        query_request = QueryRequest(
            prefetch=[
                # Sparse vector search
                PrefetchQuery(
                    query=SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist()
                    ),
                    using="sparse",
                    limit=prefetch_limit,
                    filter=filters
                ),
                # Dense vector search
                PrefetchQuery(
                    query=dense_vector.tolist(),
                    using="dense",
                    limit=prefetch_limit,
                    filter=filters
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF if fusion_method == "rrf" else Fusion.DBSF),
            limit=final_limit,
            with_payload=True,
            with_vector=False  # Don't return vectors to save bandwidth
        )
        
        # Execute search
        results = await self.client.query_points(
            collection_name=collection,
            query_request=query_request
        )
        
        return self._format_results(results)
```

### 2. Multi-Stage Search with Matryoshka Embeddings

```python
class MultiStageSearchService:
    """
    Implements multi-stage search using Matryoshka embeddings
    for efficient coarse-to-fine retrieval.
    """
    
    async def multi_stage_search(
        self,
        query: str,
        collection: str,
        stages: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-stage search with increasingly precise embeddings.
        
        Default stages:
        1. Small embedding (256d) → 1000 results
        2. Medium embedding (768d) → 100 results  
        3. Large embedding (3072d) → 10 results
        """
        
        if stages is None:
            stages = [
                {"vector_name": "embedding_small", "limit": 1000},
                {"vector_name": "embedding_medium", "limit": 100},
                {"vector_name": "embedding_large", "limit": 10}
            ]
        
        # Generate query embeddings at different dimensions
        query_embeddings = await self.generate_matryoshka_embeddings(query)
        
        # Build nested prefetch query
        query_request = self._build_nested_query(query_embeddings, stages)
        
        # Execute search
        results = await self.client.query_points(
            collection_name=collection,
            query_request=query_request
        )
        
        return self._format_results(results)
    
    def _build_nested_query(
        self,
        embeddings: Dict[str, np.ndarray],
        stages: List[Dict[str, Any]]
    ) -> QueryRequest:
        """Build nested prefetch query for multi-stage search."""
        
        # Start with innermost query
        current_query = Query(
            nearest=embeddings[stages[-1]["vector_name"]].tolist()
        )
        
        # Build nested prefetches from inside out
        for i in range(len(stages) - 2, -1, -1):
            stage = stages[i]
            current_query = PrefetchQuery(
                query=embeddings[stage["vector_name"]].tolist(),
                using=stage["vector_name"],
                limit=stage["limit"],
                prefetch=current_query if i < len(stages) - 1 else None
            )
        
        return QueryRequest(
            prefetch=current_query if len(stages) > 1 else None,
            query=current_query if len(stages) == 1 else None,
            with_payload=True
        )
```

### 3. Sparse Vector Generation

```python
from transformers import AutoTokenizer, AutoModel
import torch
import scipy.sparse as sp

class SPLADEEncoder:
    """SPLADE++ encoder for sparse vector generation."""
    
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    async def encode(self, text: str) -> SparseVector:
        """Generate sparse vector representation."""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Generate sparse activations
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply log-saturation and ReLU
            sparse_vec = torch.log1p(torch.relu(logits))
            
            # Get non-zero indices and values
            indices = torch.nonzero(sparse_vec).squeeze()
            values = sparse_vec[indices].tolist()
            
        return SparseVector(
            indices=indices.tolist(),
            values=values
        )

class BM25Encoder:
    """Alternative: BM25-based sparse encoder."""
    
    def __init__(self, analyzer, vocabulary: Dict[str, int]):
        self.analyzer = analyzer
        self.vocabulary = vocabulary
        self.idf_weights = self._compute_idf_weights()
        
    async def encode(self, text: str) -> SparseVector:
        """Generate BM25 sparse vector."""
        
        # Tokenize and analyze
        tokens = self.analyzer(text)
        
        # Compute term frequencies
        term_freqs = {}
        for token in tokens:
            if token in self.vocabulary:
                term_freqs[self.vocabulary[token]] = term_freqs.get(
                    self.vocabulary[token], 0
                ) + 1
        
        # Apply BM25 weighting
        indices = []
        values = []
        
        doc_len = len(tokens)
        avg_doc_len = 1500  # Pre-computed average
        
        for idx, freq in term_freqs.items():
            # BM25 formula
            k1, b = 1.2, 0.75
            idf = self.idf_weights.get(idx, 0)
            
            score = idf * (freq * (k1 + 1)) / (
                freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            )
            
            if score > 0:
                indices.append(idx)
                values.append(float(score))
        
        return SparseVector(indices=indices, values=values)
```

### 4. Advanced Reranking Implementation

```python
from sentence_transformers import CrossEncoder
import asyncio
from typing import List, Tuple

class RerankerService:
    """Multi-model reranking service."""
    
    def __init__(self):
        self.rerankers = {
            "bge-reranker-v2-m3": CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                max_length=512
            ),
            "ms-marco-MiniLM": CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        }
        
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        model: str = "bge-reranker-v2-m3",
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            query: Search query
            documents: List of documents with 'content' field
            model: Reranker model to use
            top_k: Number of top results to return
            batch_size: Batch size for inference
        """
        
        if not documents:
            return []
        
        reranker = self.rerankers[model]
        
        # Prepare pairs for reranking
        pairs = [
            (query, doc.get("content", doc.get("text", "")))
            for doc in documents
        ]
        
        # Score in batches for efficiency
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            
            # Run scoring in thread pool to avoid blocking
            scores = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: reranker.predict(batch)
            )
            all_scores.extend(scores)
        
        # Sort by scores and return top_k
        scored_docs = list(zip(documents, all_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Add reranking scores to results
        results = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            results.append(doc_copy)
            
        return results

class ColBERTReranker:
    """ColBERT-style multi-vector reranking."""
    
    async def rerank_with_colbert(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_encoder,
        doc_encoder,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank using ColBERT late interaction.
        
        Computes MaxSim between query and document token embeddings.
        """
        
        # Encode query to multiple vectors
        query_vecs = await query_encoder.encode_multi(query)
        
        scored_docs = []
        for doc in documents:
            # Get pre-computed document vectors or compute them
            if "colbert_vecs" in doc:
                doc_vecs = doc["colbert_vecs"]
            else:
                doc_vecs = await doc_encoder.encode_multi(doc["content"])
            
            # Compute MaxSim score
            score = self._compute_maxsim(query_vecs, doc_vecs)
            scored_docs.append((doc, score))
        
        # Sort and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = doc.copy()
            doc_copy["colbert_score"] = float(score)
            results.append(doc_copy)
            
        return results
    
    def _compute_maxsim(
        self,
        query_vecs: np.ndarray,
        doc_vecs: np.ndarray
    ) -> float:
        """Compute MaxSim score between query and document vectors."""
        
        # Compute cosine similarity matrix
        sim_matrix = np.dot(query_vecs, doc_vecs.T)
        
        # Max-pooling over document dimension
        max_sims = np.max(sim_matrix, axis=1)
        
        # Sum over query dimension
        score = np.sum(max_sims)
        
        return float(score)
```

### 5. Query Enhancement & Understanding

```python
from typing import List, Dict, Tuple
import spacy

class QueryEnhancer:
    """Enhance queries for better search results."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.acronym_db = self._load_acronym_database()
        self.synonym_db = self._load_synonym_database()
        
    async def enhance_query(
        self,
        query: str,
        expand_acronyms: bool = True,
        add_synonyms: bool = True,
        detect_intent: bool = True
    ) -> Dict[str, Any]:
        """
        Enhance query with various techniques.
        
        Returns:
            Enhanced query info including:
            - original_query
            - enhanced_query
            - detected_intent
            - expansions
        """
        
        doc = self.nlp(query)
        enhanced_parts = []
        expansions = []
        
        # Detect query intent
        intent = self._detect_intent(doc) if detect_intent else "general"
        
        # Process tokens
        for token in doc:
            # Expand acronyms
            if expand_acronyms and token.text.isupper():
                expansion = self.acronym_db.get(token.text)
                if expansion:
                    enhanced_parts.append(f"{token.text} ({expansion})")
                    expansions.append({
                        "type": "acronym",
                        "original": token.text,
                        "expansion": expansion
                    })
                    continue
            
            # Add synonyms for key terms
            if add_synonyms and token.pos_ in ["NOUN", "VERB"]:
                synonyms = self.synonym_db.get(token.lemma_, [])
                if synonyms:
                    enhanced_parts.append(token.text)
                    expansions.append({
                        "type": "synonym",
                        "original": token.text,
                        "synonyms": synonyms[:3]  # Limit synonyms
                    })
                    continue
            
            enhanced_parts.append(token.text)
        
        enhanced_query = " ".join(enhanced_parts)
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "detected_intent": intent,
            "expansions": expansions,
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "key_phrases": self._extract_key_phrases(doc)
        }
    
    def _detect_intent(self, doc) -> str:
        """Detect search intent from query."""
        
        # Simple rule-based intent detection
        text_lower = doc.text.lower()
        
        if any(word in text_lower for word in ["how to", "tutorial", "guide"]):
            return "tutorial"
        elif any(word in text_lower for word in ["what is", "define", "meaning"]):
            return "definition"
        elif any(word in text_lower for word in ["error", "issue", "problem", "fix"]):
            return "troubleshooting"
        elif any(word in text_lower for word in ["example", "sample", "demo"]):
            return "example"
        elif any(word in text_lower for word in ["api", "reference", "method"]):
            return "api_reference"
        else:
            return "general"
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key phrases from query."""
        
        phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            phrases.append(chunk.text)
        
        # Extract verb phrases
        for token in doc:
            if token.pos_ == "VERB":
                phrase_tokens = [token]
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj", "prep"]:
                        phrase_tokens.append(child)
                if len(phrase_tokens) > 1:
                    phrases.append(" ".join([t.text for t in phrase_tokens]))
        
        return list(set(phrases))  # Remove duplicates
```

### 6. Score Boosting and Result Adjustment

```python
from datetime import datetime
from typing import Dict, Any, List

class ScoreBooster:
    """Apply score boosting based on metadata and business rules."""
    
    async def apply_boosting(
        self,
        results: List[Dict[str, Any]],
        boost_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply score boosting to search results.
        
        Example boost_config:
        {
            "recency": {"enabled": True, "weight": 0.2},
            "popularity": {"enabled": True, "weight": 0.1},
            "doc_type": {
                "enabled": True,
                "weights": {"tutorial": 1.2, "api": 1.1, "guide": 1.0}
            },
            "source": {
                "enabled": True,
                "weights": {"official": 1.3, "community": 0.9}
            }
        }
        """
        
        boosted_results = []
        
        for result in results:
            original_score = result.get("score", 1.0)
            boost_multiplier = 1.0
            boost_details = {}
            
            # Recency boost
            if boost_config.get("recency", {}).get("enabled"):
                recency_boost = self._calculate_recency_boost(
                    result.get("metadata", {}).get("updated_at"),
                    boost_config["recency"]["weight"]
                )
                boost_multiplier *= recency_boost
                boost_details["recency"] = recency_boost
            
            # Popularity boost
            if boost_config.get("popularity", {}).get("enabled"):
                popularity_boost = self._calculate_popularity_boost(
                    result.get("metadata", {}).get("views", 0),
                    boost_config["popularity"]["weight"]
                )
                boost_multiplier *= popularity_boost
                boost_details["popularity"] = popularity_boost
            
            # Document type boost
            if boost_config.get("doc_type", {}).get("enabled"):
                doc_type = result.get("metadata", {}).get("type", "general")
                type_weight = boost_config["doc_type"]["weights"].get(doc_type, 1.0)
                boost_multiplier *= type_weight
                boost_details["doc_type"] = type_weight
            
            # Source authority boost
            if boost_config.get("source", {}).get("enabled"):
                source = result.get("metadata", {}).get("source", "unknown")
                source_weight = boost_config["source"]["weights"].get(source, 1.0)
                boost_multiplier *= source_weight
                boost_details["source"] = source_weight
            
            # Apply boost
            boosted_score = original_score * boost_multiplier
            
            # Create boosted result
            boosted_result = result.copy()
            boosted_result["score"] = boosted_score
            boosted_result["original_score"] = original_score
            boosted_result["boost_details"] = boost_details
            boosted_result["boost_multiplier"] = boost_multiplier
            
            boosted_results.append(boosted_result)
        
        # Re-sort by boosted scores
        boosted_results.sort(key=lambda x: x["score"], reverse=True)
        
        return boosted_results
    
    def _calculate_recency_boost(
        self,
        updated_at: str,
        weight: float
    ) -> float:
        """Calculate boost based on document recency."""
        
        if not updated_at:
            return 1.0
        
        try:
            # Parse date
            update_date = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            days_old = (datetime.now() - update_date).days
            
            # Exponential decay
            # Documents < 30 days get boost, older get penalty
            if days_old < 30:
                boost = 1 + weight * (1 - days_old / 30)
            elif days_old < 365:
                boost = 1 - weight * 0.5 * (days_old - 30) / 335
            else:
                boost = 1 - weight * 0.5
            
            return max(0.5, min(1.5, boost))  # Clamp between 0.5 and 1.5
            
        except Exception:
            return 1.0
    
    def _calculate_popularity_boost(
        self,
        views: int,
        weight: float
    ) -> float:
        """Calculate boost based on popularity metrics."""
        
        if views <= 0:
            return 1.0
        
        # Log-based boost to handle large view counts
        import math
        
        # Normalize views (assuming 10k views is "very popular")
        normalized = math.log10(views + 1) / math.log10(10000)
        boost = 1 + weight * min(1, normalized)
        
        return max(0.9, min(1.3, boost))  # Clamp between 0.9 and 1.3
```

### 7. Query Caching Strategy

```python
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class QueryCache:
    """Intelligent query result caching."""
    
    def __init__(self, redis_client, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        
    async def get_cached_results(
        self,
        query: str,
        params: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached search results."""
        
        cache_key = self._generate_cache_key(query, params)
        cached = await self.redis.get(cache_key)
        
        if cached:
            data = json.loads(cached)
            # Check if cache is still fresh
            if self._is_cache_fresh(data):
                return data["results"]
        
        return None
    
    async def cache_results(
        self,
        query: str,
        params: Dict[str, Any],
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ):
        """Cache search results with intelligent TTL."""
        
        cache_key = self._generate_cache_key(query, params)
        
        # Determine TTL based on query characteristics
        if ttl is None:
            ttl = self._calculate_ttl(query, results)
        
        cache_data = {
            "results": results,
            "cached_at": datetime.utcnow().isoformat(),
            "query": query,
            "params": params,
            "ttl": ttl
        }
        
        await self.redis.setex(
            cache_key,
            ttl,
            json.dumps(cache_data)
        )
    
    def _generate_cache_key(
        self,
        query: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate deterministic cache key."""
        
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Sort params for consistency
        sorted_params = json.dumps(params, sort_keys=True)
        
        # Create hash
        content = f"{normalized_query}:{sorted_params}"
        hash_key = hashlib.sha256(content.encode()).hexdigest()
        
        return f"search:v1:{hash_key}"
    
    def _calculate_ttl(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> int:
        """Calculate appropriate TTL based on query and results."""
        
        # Short queries might be more volatile
        if len(query.split()) <= 2:
            base_ttl = 1800  # 30 minutes
        else:
            base_ttl = 3600  # 1 hour
        
        # Popular queries can be cached longer
        if self._is_common_query(query):
            base_ttl *= 2
        
        # If few results, might want shorter cache
        if len(results) < 5:
            base_ttl = int(base_ttl * 0.5)
        
        return base_ttl
    
    def _is_common_query(self, query: str) -> bool:
        """Check if query is common/popular."""
        
        common_patterns = [
            "how to", "what is", "tutorial",
            "getting started", "installation",
            "documentation", "api reference"
        ]
        
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in common_patterns)
```

## Integration Examples

### Complete Search Pipeline

```python
class UnifiedSearchService:
    """Unified search service combining all techniques."""
    
    def __init__(self, config: SearchConfig):
        self.qdrant = AsyncQdrantClient(url=config.qdrant_url)
        self.hybrid_search = HybridSearchService(self.qdrant)
        self.reranker = RerankerService()
        self.query_enhancer = QueryEnhancer()
        self.score_booster = ScoreBooster()
        self.cache = QueryCache(redis_client, ttl=3600)
        
    async def search(
        self,
        query: str,
        options: SearchOptions = None
    ) -> SearchResponse:
        """
        Perform complete search pipeline.
        """
        
        if options is None:
            options = SearchOptions()
        
        # Check cache
        cache_key = f"{query}:{options.dict()}"
        cached_results = await self.cache.get(cache_key)
        if cached_results and not options.skip_cache:
            return SearchResponse(
                results=cached_results,
                cached=True
            )
        
        # Enhance query
        enhanced = await self.query_enhancer.enhance_query(query)
        
        # Perform hybrid search
        search_results = await self.hybrid_search.hybrid_search(
            query=enhanced["enhanced_query"],
            collection=options.collection,
            fusion_method=options.fusion_method,
            prefetch_limit=options.prefetch_limit,
            final_limit=options.pre_rerank_limit
        )
        
        # Apply reranking if enabled
        if options.enable_reranking and len(search_results) > 0:
            search_results = await self.reranker.rerank(
                query=query,
                documents=search_results,
                model=options.rerank_model,
                top_k=options.final_limit
            )
        
        # Apply score boosting
        if options.boost_config:
            search_results = await self.score_booster.apply_boosting(
                results=search_results,
                boost_config=options.boost_config
            )
        
        # Format final results
        final_results = self._format_results(
            results=search_results[:options.final_limit],
            query_info=enhanced
        )
        
        # Cache results
        await self.cache.set(cache_key, final_results)
        
        return SearchResponse(
            results=final_results,
            query_info=enhanced,
            total_found=len(search_results),
            cached=False
        )
```

## Performance Optimization Tips

### 1. Batch Processing
- Process multiple queries in parallel
- Batch embedding generation
- Use connection pooling for Qdrant

### 2. Caching Strategy
- Cache embeddings for common queries
- Cache search results with intelligent TTL
- Use Redis for distributed caching

### 3. Index Optimization
- Use appropriate HNSW parameters
- Enable quantization for large collections
- Partition data by time or category

### 4. Query Optimization
- Limit prefetch sizes based on needs
- Use filters to reduce search space
- Implement query complexity analysis

## Monitoring and Analytics

```python
class SearchAnalytics:
    """Track and analyze search performance."""
    
    async def track_search(
        self,
        query: str,
        results_count: int,
        latency_ms: float,
        cache_hit: bool,
        search_type: str
    ):
        """Track search metrics."""
        
        await self.metrics.increment(
            "search_requests_total",
            tags={
                "search_type": search_type,
                "cache_hit": str(cache_hit)
            }
        )
        
        await self.metrics.histogram(
            "search_latency_ms",
            latency_ms,
            tags={"search_type": search_type}
        )
        
        await self.metrics.gauge(
            "search_results_count",
            results_count,
            tags={"search_type": search_type}
        )
        
        # Store for analysis
        await self.store_search_event({
            "query": query,
            "results_count": results_count,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "search_type": search_type,
            "timestamp": datetime.utcnow()
        })
```

## Best Practices

1. **Always use prefetch** for hybrid search to leverage Qdrant's optimization
2. **Implement fallbacks** for when reranking models are unavailable
3. **Monitor latencies** and adjust prefetch limits accordingly
4. **Use appropriate fusion methods** - RRF for general use, DBSF for specific domains
5. **Cache aggressively** but with intelligent invalidation
6. **Profile your queries** to understand bottlenecks
7. **Use streaming** for large result sets
8. **Implement circuit breakers** for external services

This implementation provides a robust foundation for advanced search capabilities with room for customization based on specific use cases.