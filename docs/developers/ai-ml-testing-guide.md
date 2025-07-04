# AI/ML Testing Guide

> **Purpose**: Comprehensive guide for testing AI/ML components and workflows
> **Audience**: ML Engineers and Developers working on AI features
> **Last Updated**: 2025-01-04

This guide covers testing strategies, patterns, and best practices specific to AI/ML components in the AI Docs Vector DB Hybrid Scraper system.

## AI/ML Testing Overview

### Unique Challenges
- Non-deterministic outputs
- Data dependencies
- Model drift and performance degradation
- Complex pipelines with multiple components
- Large datasets and long training times

### Testing Philosophy
- Focus on properties and behaviors rather than exact outputs
- Test data quality and pipeline integrity
- Validate model performance boundaries
- Ensure reproducibility where possible

## Embedding Service Testing

### Vector Embedding Validation
```python
import numpy as np
from hypothesis import given, strategies as st

def test_embedding_dimension_consistency():
    """Test that embeddings have consistent dimensions."""
    texts = ["AI documentation", "vector database", "machine learning"]
    embeddings = [embedding_service.generate(text) for text in texts]
    
    dimensions = [len(emb) for emb in embeddings]
    assert all(dim == dimensions[0] for dim in dimensions)

def test_embedding_normalization():
    """Test that embeddings are properly normalized."""
    text = "Sample text for embedding"
    embedding = embedding_service.generate(text)
    
    # Check L2 norm is close to 1 (for normalized embeddings)
    norm = np.linalg.norm(embedding)
    assert 0.95 <= norm <= 1.05

@given(st.text(min_size=1, max_size=500))
def test_embedding_stability(text):
    """Test embedding stability across multiple calls."""
    embedding1 = embedding_service.generate(text)
    embedding2 = embedding_service.generate(text)
    
    # Embeddings should be identical for same input
    np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
```

### Semantic Similarity Testing
```python
def test_semantic_similarity_relationships():
    """Test that semantically similar texts have higher similarity."""
    similar_pairs = [
        ("AI documentation", "artificial intelligence docs"),
        ("vector database", "vector storage system"),
        ("machine learning", "ML algorithms")
    ]
    
    dissimilar_pairs = [
        ("AI documentation", "cooking recipes"),
        ("vector database", "weather forecast"),
        ("machine learning", "car maintenance")
    ]
    
    for text1, text2 in similar_pairs:
        emb1 = embedding_service.generate(text1)
        emb2 = embedding_service.generate(text2)
        similarity = cosine_similarity(emb1, emb2)
        assert similarity > 0.6, f"Similar texts should have high similarity: {text1} vs {text2}"
    
    for text1, text2 in dissimilar_pairs:
        emb1 = embedding_service.generate(text1)
        emb2 = embedding_service.generate(text2)
        similarity = cosine_similarity(emb1, emb2)
        assert similarity < 0.4, f"Dissimilar texts should have low similarity: {text1} vs {text2}"
```

## Vector Database Testing

### Search Quality Testing
```python
def test_vector_search_relevance():
    """Test that vector search returns relevant results."""
    # Setup test documents
    documents = [
        Document(content="Python machine learning tutorial", id="1"),
        Document(content="JavaScript web development guide", id="2"),
        Document(content="AI model deployment best practices", id="3"),
        Document(content="Database optimization techniques", id="4")
    ]
    
    # Index documents
    vector_db.index_documents(documents)
    
    # Search for ML-related content
    query = "machine learning python"
    results = vector_db.search(query, limit=2)
    
    # Verify most relevant document is returned
    assert len(results) >= 1
    assert results[0].id in ["1", "3"]  # ML-related documents
    assert results[0].score > 0.7

def test_search_result_ordering():
    """Test that search results are properly ordered by relevance."""
    query = "AI documentation"
    results = vector_db.search(query, limit=10)
    
    # Scores should be in descending order
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    
    # All scores should be positive
    assert all(score > 0 for score in scores)
```

### Performance Testing
```python
def test_search_performance():
    """Test search performance under load."""
    # Index a reasonable number of documents
    documents = generate_test_documents(1000)
    vector_db.index_documents(documents)
    
    # Test search latency
    import time
    query = "test search query"
    
    start_time = time.time()
    results = vector_db.search(query, limit=10)
    search_time = time.time() - start_time
    
    assert search_time < 1.0  # Should complete within 1 second
    assert len(results) > 0

def test_concurrent_search_performance():
    """Test performance under concurrent load."""
    import concurrent.futures
    import time
    
    def search_operation():
        return vector_db.search("concurrent test", limit=5)
    
    # Run 10 concurrent searches
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(search_operation) for _ in range(10)]
        results = [future.result() for future in futures]
    
    total_time = time.time() - start_time
    
    assert total_time < 5.0  # All searches should complete within 5 seconds
    assert all(len(result) > 0 for result in results)
```

## Reranking and Hybrid Search Testing

### BGE Reranker Testing
```python
def test_reranker_improves_relevance():
    """Test that reranking improves search result relevance."""
    query = "machine learning documentation"
    
    # Get initial search results
    initial_results = vector_db.search(query, limit=20)
    
    # Apply reranking
    reranked_results = reranker.rerank(query, initial_results, top_k=10)
    
    # Reranked results should have better relevance scores
    assert len(reranked_results) <= len(initial_results)
    
    # Top reranked result should have higher relevance than average initial result
    top_reranked_score = reranked_results[0].score
    avg_initial_score = sum(r.score for r in initial_results[:10]) / 10
    assert top_reranked_score >= avg_initial_score

def test_reranker_preserves_document_content():
    """Test that reranking preserves document content integrity."""
    query = "test query"
    initial_results = vector_db.search(query, limit=10)
    reranked_results = reranker.rerank(query, initial_results, top_k=5)
    
    # All reranked documents should be from initial results
    initial_ids = {r.id for r in initial_results}
    reranked_ids = {r.id for r in reranked_results}
    assert reranked_ids.issubset(initial_ids)
    
    # Document content should be unchanged
    for result in reranked_results:
        original = next(r for r in initial_results if r.id == result.id)
        assert result.content == original.content
```

### Hybrid Search Testing
```python
def test_hybrid_search_combination():
    """Test that hybrid search combines dense and sparse retrieval."""
    query = "artificial intelligence documentation"
    
    # Get results from individual methods
    dense_results = dense_retriever.search(query, limit=10)
    sparse_results = sparse_retriever.search(query, limit=10)
    hybrid_results = hybrid_search.search(query, limit=10)
    
    # Hybrid should combine both approaches
    dense_ids = {r.id for r in dense_results}
    sparse_ids = {r.id for r in sparse_results}
    hybrid_ids = {r.id for r in hybrid_results}
    
    # Hybrid results should include elements from both
    assert len(hybrid_ids & dense_ids) > 0
    assert len(hybrid_ids & sparse_ids) > 0

def test_hybrid_search_fusion_scoring():
    """Test that hybrid search properly fuses scores."""
    query = "machine learning guide"
    results = hybrid_search.search(query, limit=5)
    
    # All results should have combined scores
    for result in results:
        assert hasattr(result, 'dense_score')
        assert hasattr(result, 'sparse_score')
        assert hasattr(result, 'combined_score')
        
        # Combined score should be reasonable fusion
        assert 0 <= result.combined_score <= 1
```

## Query Enhancement Testing

### HyDE (Hypothetical Document Embeddings) Testing
```python
def test_hyde_query_enhancement():
    """Test that HyDE improves query representation."""
    original_query = "how to optimize vector search"
    
    # Generate hypothetical document
    enhanced_query = hyde_enhancer.enhance(original_query)
    
    # Enhanced query should be longer and more detailed
    assert len(enhanced_query) > len(original_query)
    assert original_query.lower() in enhanced_query.lower()
    
    # Search with enhanced query should return relevant results
    results = vector_db.search(enhanced_query, limit=5)
    assert len(results) > 0
    assert all(result.score > 0.5 for result in results)

def test_hyde_semantic_expansion():
    """Test that HyDE semantically expands queries."""
    queries_and_expected_terms = [
        ("ML model training", ["machine learning", "neural network", "algorithm"]),
        ("vector database", ["embedding", "similarity", "search"]),
        ("API documentation", ["endpoint", "request", "response"])
    ]
    
    for query, expected_terms in queries_and_expected_terms:
        enhanced = hyde_enhancer.enhance(query)
        enhanced_lower = enhanced.lower()
        
        # Should contain at least one expected term
        assert any(term in enhanced_lower for term in expected_terms)
```

## Data Quality Testing

### Document Processing Pipeline Testing
```python
def test_document_chunking_quality():
    """Test that document chunking produces quality chunks."""
    long_document = create_test_document(length=5000)  # 5k characters
    
    chunks = chunking_service.chunk_document(long_document)
    
    # Quality checks
    assert len(chunks) > 1  # Should be split into multiple chunks
    assert all(len(chunk.content) <= MAX_CHUNK_SIZE for chunk in chunks)
    assert all(len(chunk.content) >= MIN_CHUNK_SIZE for chunk in chunks[:-1])  # Except last
    
    # Overlap check
    for i in range(len(chunks) - 1):
        overlap = find_overlap(chunks[i].content, chunks[i + 1].content)
        assert len(overlap) >= MIN_OVERLAP_SIZE

def test_metadata_preservation():
    """Test that document metadata is preserved through processing."""
    original_doc = Document(
        content="Test content",
        metadata={"source": "test.pdf", "author": "Test Author", "date": "2025-01-04"}
    )
    
    processed_doc = processing_pipeline.process(original_doc)
    
    # Metadata should be preserved
    assert processed_doc.metadata["source"] == original_doc.metadata["source"]
    assert processed_doc.metadata["author"] == original_doc.metadata["author"]
    assert processed_doc.metadata["date"] == original_doc.metadata["date"]
```

### Input Validation Testing
```python
def test_malformed_input_handling():
    """Test handling of malformed or edge case inputs."""
    edge_cases = [
        "",  # Empty string
        " " * 1000,  # Only whitespace
        "a" * 10000,  # Very long single word
        "🎉🎊🎈" * 100,  # Unicode characters
        "\n\t\r" * 50,  # Special characters
        None,  # None input
    ]
    
    for case in edge_cases:
        try:
            if case is None:
                with pytest.raises((ValueError, TypeError)):
                    embedding_service.generate(case)
            else:
                result = embedding_service.generate(case)
                if case.strip():  # Non-empty after stripping
                    assert len(result) == EMBEDDING_DIMENSION
                else:  # Empty or whitespace-only
                    assert result is None or len(result) == EMBEDDING_DIMENSION
        except Exception as e:
            pytest.fail(f"Unexpected exception for input {repr(case)}: {e}")
```

## Model Performance Monitoring

### Performance Degradation Detection
```python
def test_embedding_quality_regression():
    """Test for embedding quality regression over time."""
    # Use a set of reference queries and expected similar documents
    reference_tests = [
        {
            "query": "machine learning tutorial",
            "expected_docs": ["ml_guide.md", "ai_tutorial.md"],
            "min_similarity": 0.7
        },
        {
            "query": "API documentation",
            "expected_docs": ["api_reference.md", "endpoint_guide.md"],
            "min_similarity": 0.6
        }
    ]
    
    for test_case in reference_tests:
        query_embedding = embedding_service.generate(test_case["query"])
        
        for doc_id in test_case["expected_docs"]:
            doc = get_test_document(doc_id)
            doc_embedding = embedding_service.generate(doc.content)
            
            similarity = cosine_similarity(query_embedding, doc_embedding)
            assert similarity >= test_case["min_similarity"], \
                f"Quality regression detected: {test_case['query']} vs {doc_id} similarity: {similarity}"

def test_search_latency_regression():
    """Test for search latency regression."""
    # Baseline performance expectations
    EXPECTED_LATENCIES = {
        "simple_query": 0.1,  # 100ms
        "complex_query": 0.5,  # 500ms
        "bulk_search": 2.0,   # 2 seconds for 100 queries
    }
    
    import time
    
    # Simple query test
    start = time.time()
    vector_db.search("test query", limit=10)
    simple_latency = time.time() - start
    assert simple_latency < EXPECTED_LATENCIES["simple_query"]
    
    # Complex query test
    complex_query = "artificial intelligence machine learning deep learning neural networks"
    start = time.time()
    vector_db.search(complex_query, limit=50)
    complex_latency = time.time() - start
    assert complex_latency < EXPECTED_LATENCIES["complex_query"]
```

## Integration Testing

### End-to-End ML Pipeline Testing
```python
@pytest.mark.integration
async def test_full_ml_pipeline():
    """Test complete ML pipeline from document ingestion to search results."""
    # 1. Document ingestion
    test_document = create_test_document("AI documentation best practices")
    doc_id = await document_service.ingest_document(test_document)
    
    # 2. Wait for processing
    await asyncio.sleep(2)  # Allow time for async processing
    
    # 3. Verify document is indexed
    indexed_doc = await vector_db.get_document(doc_id)
    assert indexed_doc is not None
    assert indexed_doc.embedding is not None
    
    # 4. Test search retrieval
    results = await search_service.search("AI documentation", limit=10)
    doc_ids = [r.id for r in results]
    assert doc_id in doc_ids
    
    # 5. Test reranking
    reranked = await reranking_service.rerank("best practices", results[:5])
    assert len(reranked) > 0

@pytest.mark.integration
def test_ml_model_versioning():
    """Test that ML model versions are properly managed."""
    current_version = embedding_service.get_model_version()
    assert current_version is not None
    
    # Test model consistency
    test_input = "consistency test"
    embedding1 = embedding_service.generate(test_input)
    embedding2 = embedding_service.generate(test_input)
    
    np.testing.assert_array_almost_equal(embedding1, embedding2)
```

## Test Data Management

### Synthetic Data Generation
```python
def generate_test_documents(count: int) -> List[Document]:
    """Generate synthetic documents for testing."""
    from faker import Faker
    fake = Faker()
    
    documents = []
    topics = ["AI", "machine learning", "database", "API", "documentation"]
    
    for i in range(count):
        topic = fake.random_element(topics)
        content = f"{topic} {fake.text(max_nb_chars=500)}"
        
        doc = Document(
            id=f"test_doc_{i}",
            content=content,
            metadata={
                "topic": topic,
                "created_at": fake.date_time().isoformat(),
                "synthetic": True
            }
        )
        documents.append(doc)
    
    return documents

def create_test_embedding_dataset():
    """Create a labeled dataset for embedding quality testing."""
    similar_pairs = [
        ("machine learning algorithms", "ML model training"),
        ("vector database search", "similarity-based retrieval"),
        ("API documentation", "endpoint reference guide")
    ]
    
    dissimilar_pairs = [
        ("machine learning", "cooking recipes"),
        ("vector database", "weather forecast"),
        ("API documentation", "car repair manual")
    ]
    
    return {
        "similar_pairs": similar_pairs,
        "dissimilar_pairs": dissimilar_pairs
    }
```

## Continuous ML Testing

### Automated Quality Gates
```python
def test_ml_quality_gates():
    """Automated quality gates for ML components."""
    # Embedding quality gate
    test_embedding = embedding_service.generate("quality gate test")
    assert len(test_embedding) == EXPECTED_DIMENSION
    assert not np.allclose(test_embedding, 0)  # Not all zeros
    
    # Search quality gate
    results = vector_db.search("quality test", limit=5)
    assert len(results) >= 1
    assert all(result.score > 0.1 for result in results)
    
    # Performance quality gate
    import time
    start = time.time()
    for _ in range(10):
        embedding_service.generate("performance test")
    avg_time = (time.time() - start) / 10
    assert avg_time < 0.1  # 100ms average per embedding
```

## Resources and Tools

### Testing Libraries
- **pytest**: Primary testing framework
- **hypothesis**: Property-based testing for ML components
- **numpy.testing**: Numerical array comparisons
- **scikit-learn.metrics**: Similarity and evaluation metrics
- **respx**: HTTP mocking for external ML APIs

### ML Testing Best Practices
1. Test data quality before model quality
2. Use property-based testing for robustness
3. Monitor for model drift and performance degradation
4. Test edge cases and adversarial inputs
5. Validate reproducibility where possible
6. Separate deterministic and non-deterministic components

### Performance Benchmarking
```bash
# Run ML performance benchmarks
pytest tests/ml/ --benchmark-only --benchmark-sort=mean

# Generate ML test coverage report
pytest tests/ml/ --cov=src/services/ml --cov-report=html
```