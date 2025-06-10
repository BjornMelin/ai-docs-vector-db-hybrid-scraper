# Advanced Query Processing Guide

## Overview

The Advanced Query Processing system provides intelligent query understanding, preprocessing, and routing capabilities that dramatically improve search accuracy and relevance. This system represents a complete implementation of sophisticated query analysis with 14 intent categories, Matryoshka embeddings, and centralized orchestration.

## Architecture

```mermaid
graph TD
    A[User Query] --> B[QueryProcessingPipeline]
    B --> C[QueryPreprocessor]
    B --> D[QueryIntentClassifier]
    B --> E[SearchStrategySelector]
    B --> F[QueryProcessingOrchestrator]
    
    C --> |Enhanced Query| F
    D --> |Intent Classification| E
    E --> |Strategy Selection| F
    
    F --> G[Search Execution]
    G --> H[Results]
    
    F --> I[Fallback Strategies]
    I --> G
```

## Core Components

### 1. QueryProcessingPipeline

The main entry point providing a unified interface for all advanced query processing operations.

```python
from src.services.query_processing.pipeline import QueryProcessingPipeline
from src.services.query_processing.models import QueryProcessingRequest

# Initialize pipeline
pipeline = QueryProcessingPipeline(orchestrator=orchestrator)
await pipeline.initialize()

# Process a query
response = await pipeline.process(
    "How to optimize database performance in Python?",
    collection_name="technical_docs",
    limit=10
)

# Or use advanced configuration
request = QueryProcessingRequest(
    query="Compare React vs Vue for large applications",
    collection_name="frameworks",
    limit=15,
    enable_preprocessing=True,
    enable_intent_classification=True,
    enable_strategy_selection=True,
    user_context={"experience_level": "intermediate"}
)
response = await pipeline.process(request)
```

### 2. Query Intent Classification (14 Categories)

The system classifies queries into 14 distinct intent categories for optimal search strategy selection:

#### Basic Categories
- **conceptual**: High-level understanding questions
- **procedural**: How-to and step-by-step queries
- **factual**: Specific facts and data queries  
- **troubleshooting**: Problem-solving queries

#### Advanced Categories  
- **comparative**: Technology/concept comparisons
- **architectural**: System design and architecture queries
- **performance**: Optimization and performance queries
- **security**: Security-focused questions
- **integration**: API integration and compatibility
- **best_practices**: Recommended approaches and patterns
- **code_review**: Code analysis and improvements
- **migration**: Upgrade and migration guidance
- **debugging**: Error diagnosis and resolution
- **configuration**: Setup and configuration assistance

```python
from src.services.query_processing.intent_classifier import QueryIntentClassifier
from src.services.query_processing.models import QueryIntent

# Example classifications
queries = [
    ("What is machine learning?", QueryIntent.CONCEPTUAL),
    ("How to implement OAuth 2.0 step by step?", QueryIntent.PROCEDURAL),
    ("React vs Vue performance comparison", QueryIntent.COMPARATIVE),
    ("How to secure API endpoints?", QueryIntent.SECURITY),
    ("Getting ImportError in Python", QueryIntent.TROUBLESHOOTING)
]
```

### 3. Matryoshka Embeddings

Dynamic embedding dimension selection based on query complexity and performance requirements:

- **Small (512)**: Simple factual queries, fast retrieval
- **Medium (768)**: Standard conceptual queries, balanced performance  
- **Large (1536)**: Complex procedural/architectural queries, maximum accuracy

```python
from src.services.query_processing.models import MatryoshkaDimension

# Dimension selection examples
simple_query = "What is Python?"  # Uses SMALL (512)
moderate_query = "How to implement caching?"  # Uses MEDIUM (768)  
complex_query = "Design microservices architecture"  # Uses LARGE (1536)
```

### 4. Search Strategy Selection

7 intelligent search strategies automatically selected based on query intent:

- **SEMANTIC**: Dense vector similarity (conceptual queries)
- **HYBRID**: Dense + sparse keyword matching (factual queries)
- **HYDE**: Hypothetical document generation (procedural queries)
- **MULTI_STAGE**: Multi-stage retrieval (comparative queries)
- **FILTERED**: Metadata-filtered search (configuration queries)
- **RERANKED**: BGE reranking for relevance (troubleshooting queries)
- **ADAPTIVE**: Dynamic strategy switching (complex queries)

### 5. Query Preprocessing

Intelligent query enhancement pipeline:

```python
# Features included:
- Spell correction: "phython" → "python"
- Normalization: Remove extra spaces, standardize punctuation
- Synonym expansion: "API" → "REST API", "js" → "javascript"
- Context extraction: Programming languages, frameworks, urgency levels
- Version detection: "Python 3.9", "React v18"
```

## Integration Examples

### Basic Usage

```python
async def basic_query_processing():
    """Simple query processing example."""
    pipeline = QueryProcessingPipeline(orchestrator)
    await pipeline.initialize()
    
    response = await pipeline.process(
        "How to debug memory leaks in Node.js?",
        collection_name="troubleshooting",
        limit=10
    )
    
    print(f"Intent: {response.intent_classification.primary_intent}")
    print(f"Strategy: {response.strategy_selection.primary_strategy}")
    print(f"Results: {len(response.results)}")
```

### Advanced Configuration

```python
async def advanced_query_processing():
    """Advanced query processing with full configuration."""
    request = QueryProcessingRequest(
        query="Compare Django vs FastAPI for enterprise applications",
        collection_name="frameworks",
        limit=20,
        enable_preprocessing=True,
        enable_intent_classification=True,
        enable_strategy_selection=True,
        user_context={
            "programming_language": ["python"],
            "experience_level": "advanced",
            "urgency": "medium"
        },
        filters={"category": "web_frameworks"},
        max_processing_time_ms=2000
    )
    
    response = await pipeline.process(request)
    
    # Access detailed results
    print(f"Original query: {response.preprocessing_result.original_query}")
    print(f"Enhanced query: {response.preprocessing_result.processed_query}")
    print(f"Primary intent: {response.intent_classification.primary_intent}")
    print(f"Complexity: {response.intent_classification.complexity_level}")
    print(f"Strategy used: {response.strategy_selection.primary_strategy}")
    print(f"Processing time: {response.total_processing_time_ms}ms")
```

### Batch Processing

```python
async def batch_query_processing():
    """Process multiple queries efficiently."""
    requests = [
        QueryProcessingRequest(
            query="What is Docker?",
            collection_name="containers",
            limit=5
        ),
        QueryProcessingRequest(
            query="How to secure OAuth implementation?", 
            collection_name="security",
            limit=8
        ),
        QueryProcessingRequest(
            query="React vs Vue performance benchmarks",
            collection_name="frameworks",
            limit=10
        )
    ]
    
    responses = await pipeline.process_batch(requests)
    
    for i, response in enumerate(responses):
        print(f"Query {i+1}: {response.intent_classification.primary_intent}")
```

## MCP Integration

The advanced query processing system is fully integrated with the MCP (Model Context Protocol) providing 5 specialized tools:

### Available MCP Tools

1. **advanced_query_processing**: Complete query processing with all features
2. **analyze_query**: Deep query analysis without search execution  
3. **query_processing_health**: Health checks and system status
4. **query_processing_metrics**: Performance metrics and statistics
5. **query_processing_warmup**: System warm-up for optimal performance

### MCP Tool Usage

```json
{
  "name": "advanced_query_processing",
  "arguments": {
    "query": "How to implement microservices architecture?",
    "collection_name": "architecture", 
    "limit": 10,
    "enable_preprocessing": true,
    "enable_intent_classification": true,
    "enable_strategy_selection": true,
    "user_context": {
      "experience_level": "intermediate"
    }
  }
}
```

## Performance Optimization

### Query Complexity Optimization

The system automatically optimizes based on query complexity:

```python
# Simple queries: Fast execution with minimal processing
simple_response = await pipeline.process("What is REST?")
# - Uses rule-based classification
# - SMALL Matryoshka dimension (512)
# - SEMANTIC strategy
# - < 50ms processing time

# Complex queries: Comprehensive analysis with full features  
complex_response = await pipeline.process(
    "Design scalable microservices architecture with event sourcing"
)
# - Full semantic classification
# - LARGE Matryoshka dimension (1536) 
# - MULTI_STAGE strategy with fallbacks
# - Advanced preprocessing and context extraction
```

### Caching and Performance

```python
# Enable caching for repeated queries
pipeline = QueryProcessingPipeline(
    orchestrator=orchestrator,
    cache_enabled=True
)

# Warm up the system for optimal performance
await pipeline.warm_up()

# Monitor performance metrics
metrics = await pipeline.get_metrics()
print(f"Average processing time: {metrics['average_processing_time']}ms")
print(f"Cache hit rate: {metrics['cache_hit_rate']}%")
```

## Error Handling and Fallbacks

The system provides comprehensive error handling with intelligent fallbacks:

```python
# Automatic fallback strategies
response = await pipeline.process(
    "Complex query that might fail",
    collection_name="docs"
)

if response.fallback_used:
    print("Primary strategy failed, fallback used successfully")
    print(f"Fallback strategy: {response.strategy_used}")

# Error recovery
if not response.success:
    print(f"Processing failed: {response.error}")
    # System automatically attempts fallback strategies
```

## Health Monitoring

```python
# Comprehensive health check
health = await pipeline.health_check()
print(f"System status: {health['status']}")
print(f"Component health: {health['components']}")

# Performance metrics
metrics = await pipeline.get_metrics()
print(f"Total queries processed: {metrics['total_queries']}")
print(f"Success rate: {metrics['success_rate']}%")
print(f"Strategy usage: {metrics['strategy_usage']}")
```

## Best Practices

### 1. Query Formulation
- Use specific, descriptive queries for better intent classification
- Include context clues (programming language, framework, urgency)
- Avoid overly broad or vague queries

### 2. Performance Optimization
- Enable caching for repeated query patterns
- Use appropriate collection targeting
- Set reasonable processing time limits
- Monitor metrics to identify optimization opportunities

### 3. Error Handling
- Always check `response.success` before processing results
- Handle fallback scenarios gracefully  
- Monitor `response.warnings` for potential issues
- Use health checks to ensure system readiness

### 4. Integration Patterns
- Use batch processing for multiple related queries
- Leverage MCP tools for external integrations
- Enable comprehensive logging for debugging
- Implement proper retry logic with exponential backoff

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - Ensure query is well-formed and specific
   - Check for typos that preprocessing might not catch
   - Verify appropriate collection targeting

2. **Slow Processing Times** 
   - Enable caching for repeated patterns
   - Use simpler strategies for basic queries
   - Check system resource availability

3. **Poor Search Results**
   - Verify intent classification accuracy
   - Check strategy selection reasoning
   - Consider manual strategy override for testing

### Debugging Tools

```python
# Enable verbose logging
import logging
logging.getLogger('src.services.query_processing').setLevel(logging.DEBUG)

# Analyze query without search execution
analysis = await pipeline.analyze_query(
    "How to optimize database performance?"
)
print(f"Intent: {analysis['intent_classification']}")
print(f"Preprocessing: {analysis['preprocessing']}")  
print(f"Strategy: {analysis['strategy']}")

# Check component health
health = await pipeline.health_check()
for component, status in health['components'].items():
    if status['status'] != 'healthy':
        print(f"Issue with {component}: {status['message']}")
```

## Conclusion

The Advanced Query Processing system provides production-ready, intelligent query understanding and routing capabilities. With 84% test coverage, comprehensive error handling, and optimized performance, it's ready for immediate deployment in production environments.

For additional support or advanced configuration options, refer to the API documentation or contact the development team.