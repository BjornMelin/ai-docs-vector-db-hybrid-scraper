# Content Intelligence Service

The Content Intelligence Service provides AI-powered content analysis, quality assessment, and adaptive extraction recommendations to optimize web scraping and content processing workflows.

## Overview

The Content Intelligence Service uses local AI models and advanced heuristics to automatically:

- **Classify Content Types**: Semantic analysis of content into categories (documentation, code, FAQ, tutorials, etc.)
- **Assess Quality**: Multi-metric scoring for completeness, relevance, confidence, and structure
- **Extract Metadata**: Automatic enrichment from content and HTML elements
- **Recommend Adaptations**: Site-specific optimization strategies for improved extraction quality
- **Detect Duplicates**: Similarity analysis to identify redundant content

## Key Features

### Content Classification

Automatically categorizes content using AI-powered semantic analysis:

- **Documentation**: User guides, manuals, API documentation
- **Code**: Source code, examples, code snippets
- **FAQ**: Frequently asked questions and answers
- **Tutorial**: Step-by-step instructions and guides
- **Reference**: API references, specifications, technical docs
- **Blog**: Blog posts, articles, news content
- **Forum**: Discussion threads, community posts

### Quality Assessment

Multi-dimensional quality scoring across:

- **Completeness**: Content length and structural adequacy
- **Relevance**: Alignment with query context (if provided)
- **Confidence**: Extraction reliability indicators
- **Freshness**: Content recency and timestamp analysis
- **Structure**: Organization, headings, formatting quality
- **Readability**: Sentence complexity and vocabulary assessment
- **Duplicate Similarity**: Comparison with existing content

### Metadata Enrichment

Comprehensive metadata extraction including:

- **Basic**: Title, description, author, language, charset
- **Temporal**: Published date, last modified, crawl timestamp
- **Content**: Word count, paragraph count, links, images
- **Semantic**: Tags, keywords, entities, topics
- **Technical**: Content hash, extraction method, load time
- **Hierarchy**: Breadcrumbs, parent/related URLs
- **Structured**: Schema.org types, JSON-LD data

### Site-Specific Adaptations

Intelligent optimization recommendations:

- **Extract Main Content**: Focus on primary content areas
- **Follow Schema**: Use structured data patterns
- **Detect Patterns**: Analyze site-specific patterns
- **Handle Dynamic**: Execute JavaScript for SPA content
- **Bypass Navigation**: Skip non-content elements

## Usage Examples

### Basic Content Analysis

```python
from src.mcp_tools.models.requests import ContentIntelligenceAnalysisRequest

# Create analysis request
request = ContentIntelligenceAnalysisRequest(
    url="https://example.com/docs/guide",
    content="Complete guide to implementing authentication...",
    title="Authentication Guide",
    enable_classification=True,
    enable_quality_assessment=True,
    enable_metadata_extraction=True
)

# Perform analysis
result = await analyze_content_intelligence(request)

if result.success:
    classification = result.enriched_content.classification
    quality = result.enriched_content.quality_score
    metadata = result.enriched_content.metadata
    
    print(f"Content Type: {classification.primary_type}")
    print(f"Quality Score: {quality.overall_score:.2f}")
    print(f"Word Count: {metadata.word_count}")
```

### Content Type Classification

```python
from src.mcp_tools.models.requests import ContentIntelligenceClassificationRequest

# Classify content type
request = ContentIntelligenceClassificationRequest(
    url="https://api.example.com/docs",
    content="API endpoint documentation with examples...",
    title="REST API Reference"
)

classification = await classify_content_type(request)

print(f"Primary Type: {classification.primary_type}")
print(f"Confidence: {classification.confidence_scores[classification.primary_type]:.2f}")
print(f"Reasoning: {classification.classification_reasoning}")
```

### Quality Assessment

```python
from src.mcp_tools.models.requests import ContentIntelligenceQualityRequest

# Assess content quality
request = ContentIntelligenceQualityRequest(
    content="Tutorial content with step-by-step instructions...",
    confidence_threshold=0.7,
    query_context="python tutorial authentication"
)

quality = await assess_content_quality(request)

print(f"Overall Score: {quality.overall_score:.2f}")
print(f"Meets Threshold: {quality.meets_threshold}")
print(f"Issues: {', '.join(quality.quality_issues)}")
print(f"Suggestions: {', '.join(quality.improvement_suggestions)}")
```

### Site-Specific Optimization

```python
# Get adaptation recommendations
recommendations = await get_adaptation_recommendations(
    url="https://github.com/user/repo",
    content_patterns=["markdown-body", "readme"],
    quality_issues=["incomplete_extraction", "navigation_noise"]
)

for rec in recommendations:
    print(f"Strategy: {rec['strategy']}")
    print(f"Confidence: {rec['confidence']:.2f}")
    print(f"Description: {rec['description']}")
```

## Configuration

### Service Settings

```python
# Configure Content Intelligence Service
content_service_config = {
    "confidence_threshold": 0.6,
    "enable_local_models": True,
    "cache_results": True,
    "similarity_threshold": 0.85,
    "min_content_length": 50
}
```

### Quality Thresholds

```python
# Customize quality assessment thresholds
quality_config = {
    "completeness_threshold": 0.5,
    "relevance_threshold": 0.6,
    "confidence_threshold": 0.7,
    "structure_threshold": 0.4,
    "readability_threshold": 0.5
}
```

## Performance Monitoring

### Service Metrics

```python
# Get performance metrics
metrics = await get_content_intelligence_metrics()

print(f"Total Analyses: {metrics['total_analyses']}")
print(f"Average Processing Time: {metrics['average_processing_time_ms']:.1f}ms")
print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
print(f"Service Available: {metrics['service_available']}")
```

### Optimization Tips

1. **Enable Caching**: Reuse analysis results for similar content
2. **Batch Processing**: Group similar content for efficient analysis
3. **Quality Thresholds**: Adjust thresholds based on your content requirements
4. **Local Models**: Use local models to reduce API dependencies
5. **Selective Analysis**: Enable only needed analysis components

## Integration with MCP Tools

The Content Intelligence Service integrates seamlessly with MCP tools:

- **analyze_content_intelligence**: Comprehensive content analysis
- **classify_content_type**: Content type classification only
- **assess_content_quality**: Quality assessment only  
- **extract_content_metadata**: Metadata extraction only
- **get_adaptation_recommendations**: Site-specific optimization
- **get_content_intelligence_metrics**: Performance monitoring

## Best Practices

### Content Analysis
- Always provide URL context for better classification
- Include raw HTML when available for richer metadata extraction
- Set appropriate confidence thresholds for your use case
- Monitor quality scores to identify content issues

### Quality Assessment
- Use query context to improve relevance scoring
- Review quality issues and suggestions for optimization opportunities
- Establish quality baselines for different content types
- Track quality trends over time

### Site Optimization
- Apply adaptation recommendations incrementally
- Test optimization strategies on sample content first
- Monitor extraction quality improvements after applying adaptations
- Document successful patterns for reuse

## Troubleshooting

### Common Issues

**Low Quality Scores**
- Check content length and structure
- Verify extraction completeness
- Review quality threshold settings
- Consider site-specific adaptations

**Classification Errors**
- Provide more context (URL, title)
- Check content language and encoding
- Verify content is substantial enough for analysis
- Review classification confidence scores

**Performance Issues**
- Enable result caching for repeated analyses
- Use batch processing for multiple items
- Monitor service metrics and resource usage
- Consider adjusting analysis complexity

For additional support and advanced configuration options, see the [Developer API Reference](../developers/api-reference.md).