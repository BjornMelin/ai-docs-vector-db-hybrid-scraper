---
title: Examples And Recipes
audience: users
status: active
owner: product-education
last_reviewed: 2025-03-13
---

## Examples & Recipes

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: Real-world use cases and practical examples  
> **Audience**: Users looking for specific implementation patterns

Learn through practical examples! This guide provides ready-to-use recipes for common scenarios,
showing you exactly how to accomplish real-world tasks with our system.

## 🚀 Quick Recipes

### Recipe 1: Search Technical Documentation

**Goal**: Find specific information across technical docs quickly

```bash
# Search for configuration information
mcp search --query "configure OpenAI API key environment variables"

# Find troubleshooting guides
mcp search --query "fix database connection timeout errors"

# Look for best practices
mcp search --query "production deployment checklist"
```

**When to use**: Research, problem-solving, learning new technologies

### Recipe 2: Monitor Website Changes

**Goal**: Track updates to important websites automatically

```bash
# Scrape and save content for comparison
mcp scrape --url "https://docs.example.com/changelog" --save-as "changelog"

# Later, compare with new version
mcp scrape --url "https://docs.example.com/changelog" --compare-with "changelog"
```

**When to use**: Staying updated with documentation changes, competitive monitoring

### Recipe 3: Build a Knowledge Base

**Goal**: Create searchable repository from multiple sources

```bash
# Scrape documentation sites
mcp scrape --url "https://docs.framework.com" --auto-index

# Add local documents
mcp add-documents --path "./company-docs" --collection "internal-knowledge"

# Search across everything with federated search
mcp search --query "authentication best practices" --collections "all"

# Advanced federated search with weighting
mcp search --query "production deployment" --federated \
  --collections "docs,examples,tutorials" \
  --weights "docs=0.6,examples=0.8,tutorials=0.4"
```

**When to use**: Team knowledge management, research projects, documentation consolidation

### Recipe 4: Cross-Collection Research

**Goal**: Find comprehensive information across multiple specialized collections

```bash
# Set up specialized collections
mcp scrape --url "https://api-docs.com" --collection "api-reference"
mcp scrape --url "https://tutorials.com" --collection "tutorials"
mcp scrape --url "https://examples.com" --collection "code-examples"

# Federated search with smart collection selection
mcp search --query "payment gateway integration" --federated \
  --strategy "smart_routing" --scope "comprehensive"

# Performance-based federated search
mcp search --query "database optimization" --federated \
  --strategy "performance_based" --max-collections 3
```

**When to use**: Complex research requiring multiple perspectives, comprehensive analysis

## 📚 Content Research Examples

### Academic Research

**Scenario**: Researching AI/ML papers and documentation

#### **Step 1: Gather Sources**

```bash
# Scrape research-oriented sites
mcp scrape --url "https://arxiv.org/abs/2301.00001" --extract-citations
mcp scrape --url "https://papers.nips.cc/paper/123" --save-metadata
```

#### **Step 2: Search for Concepts**

```bash
# Find related concepts across sources
mcp search --query "transformer architecture attention mechanisms"
mcp search --query "vector database similarity search algorithms"
```

#### **Step 3: Cross-Reference Findings**

```bash
# Search for specific techniques mentioned in papers
mcp search --query "HNSW algorithm parameter tuning" --include-metadata
```

### Competitive Analysis

**Scenario**: Understanding competitors' documentation and features

**Information Gathering**:

```bash
# Collect competitor documentation
mcp scrape --url "https://competitor-a.com/docs" --collection "comp-analysis"
mcp scrape --url "https://competitor-b.com/features" --collection "comp-analysis"

# Search for specific capabilities
mcp search --query "pricing model comparison" --collection "comp-analysis"
mcp search --query "API rate limits" --collection "comp-analysis"
```

**Analysis Queries**:

```bash
# Compare features across competitors using federated search
mcp search --query "authentication methods supported" --federated \
  --collections "comp-analysis" --strategy "content_based"
mcp search --query "enterprise features availability" --federated \
  --collections "comp-analysis" --merging "diversity_optimized"
mcp search --query "integration capabilities" --federated \
  --collections "comp-analysis" --scope "comprehensive"
```

### Product Research

**Scenario**: Evaluating tools and technologies for adoption

**Research Process**:

```bash
# Gather comprehensive information
mcp scrape --url "https://tool.com/docs" --extract-all
mcp scrape --url "https://tool.com/pricing" --save-metadata

# Research specific aspects
mcp search --query "setup requirements installation guide"
mcp search --query "performance benchmarks scaling limits"
mcp search --query "support options community resources"
```

## 🛠️ Development Support Examples

### API Documentation Research

**Scenario**: Understanding how to integrate with external APIs

**Discovery Phase**:

```bash
# Scrape API documentation
mcp scrape --url "https://api.service.com/docs" --extract-endpoints
mcp scrape --url "https://api.service.com/examples" --save-code-samples

# Search for specific functionality
mcp search --query "authentication OAuth2 implementation"
mcp search --query "rate limiting headers response codes"
```

**Implementation Support**:

```bash
# Find code examples and patterns
mcp search --query "Python client library usage examples"
mcp search --query "error handling best practices"
mcp search --query "webhook configuration setup"
```

### Technology Evaluation

**Scenario**: Choosing between different technical solutions

**Comparison Research**:

```bash
# Gather information on alternatives
mcp scrape --url "https://solution-a.com/docs" --collection "evaluation"
mcp scrape --url "https://solution-b.com/docs" --collection "evaluation"

# Compare specific aspects using federated search for comprehensive analysis
mcp search --query "performance characteristics throughput" --federated \
  --collections "evaluation" --merging "score_based" --scope "comprehensive"
mcp search --query "maintenance requirements support" --federated \
  --collections "evaluation" --strategy "smart_routing"
mcp search --query "cost structure pricing model" --federated \
  --collections "evaluation" --merging "temporal" --enable-dedup
```

### Problem-Solving Workflows

**Scenario**: Debugging issues and finding solutions

**Issue Research**:

```bash
# Search for similar problems
mcp search --query "connection timeout SSL certificate error"
mcp search --query "memory leak Python multiprocessing"

# Find solution patterns
mcp search --query "troubleshooting steps systematic approach"
mcp search --query "debugging tools configuration"
```

## 📊 Business Intelligence Examples

### Market Research

**Scenario**: Understanding market trends and opportunities

**Data Collection**:

```bash
# Gather industry reports and analyses
mcp scrape --url "https://research-firm.com/reports" --extract-insights
mcp scrape --url "https://industry-blog.com/trends" --save-metadata

# Search for specific trends
mcp search --query "market growth predictions 2024"
mcp search --query "emerging technologies adoption rates"
```

### Customer Support Enhancement

**Scenario**: Building comprehensive support knowledge base

**Content Gathering**:

```bash
# Collect support documentation
mcp scrape --url "https://support.product.com" --collection "support-kb"
mcp add-documents --path "./support-tickets" --collection "support-kb"

# Search for solution patterns
mcp search --query "common customer issues solutions"
mcp search --query "escalation procedures best practices"
```

### Content Strategy Research

**Scenario**: Understanding content gaps and opportunities

**Competitive Content Analysis**:

```bash
# Analyze competitors' content
mcp scrape --url "https://competitor.com/blog" --extract-topics
mcp scrape --url "https://competitor.com/resources" --save-metadata

# Identify content themes
mcp search --query "content topics popular themes"
mcp search --query "content gaps opportunity areas"
```

## 🔍 Advanced Filtering Examples

### Temporal Filtering for Time-Sensitive Content

**Scenario**: Finding recent information and tracking content freshness

```bash
# Find content updated in the last month
mcp search --query "API changes" --temporal-filter \
  --date-range "2024-12-01:2024-12-31" --freshness-threshold 0.8

# Search for the most recent deployment guides
mcp search --query "deployment guide" --temporal-filter \
  --freshness-decay 0.1 --prioritize-recent

# Track content changes over time
mcp search --query "security updates" --temporal-filter \
  --date-range "2024-01-01:now" --group-by-month
```

### Content Type and Semantic Filtering

**Scenario**: Finding specific types of content with precision

```bash
# Search only API reference documents
mcp search --query "authentication methods" --content-filter \
  --document-types "api_reference,technical_spec"

# Find code examples and implementations
mcp search --query "database connection" --content-filter \
  --semantic-categories "code_example,implementation" \
  --confidence-threshold 0.7

# Filter by content complexity level
mcp search --query "getting started" --content-filter \
  --semantic-categories "beginner,tutorial" --exclude-advanced
```

### Complex Metadata Filtering

**Scenario**: Precise filtering using document metadata with boolean logic

```bash
# Find Python web framework content excluding deprecated
mcp search --query "web framework tutorial" --metadata-filter \
  --conditions "language=python AND framework IN (django,fastapi,flask) AND NOT deprecated=true"

# Search for recent, high-quality content
mcp search --query "machine learning guide" --metadata-filter \
  --conditions "quality_score>=0.8 AND last_updated>2024-06-01 AND language=en"

# Complex enterprise content filtering
mcp search --query "deployment strategies" --metadata-filter \
  --conditions "(category=enterprise OR category=production) AND security_level>=medium"
```

### Similarity Threshold Optimization

**Scenario**: Fine-tuning search precision for different use cases

```bash
# High precision search for exact matches
mcp search --query "specific error message" --similarity-threshold 0.9 \
  --precision-mode

# Broader exploratory search
mcp search --query "general concepts" --similarity-threshold 0.6 \
  --discovery-mode --adaptive-threshold

# Dynamic threshold based on query complexity
mcp search --query "complex architectural patterns" \
  --adaptive-threshold --complexity-aware --target-results 15
```

### Filter Composition and Advanced Combinations

**Scenario**: Combining multiple filter types for precise content discovery

```bash
# Comprehensive filtered search combining all filter types
mcp search --query "production deployment best practices" \
  --temporal-filter --date-range "2024-01-01:now" \
  --content-filter --document-types "guide,best_practice" \
  --metadata-filter --conditions "environment=production AND reviewed=true" \
  --similarity-threshold 0.75 \
  --compose-filters "AND"

# Alternative content discovery with OR composition
mcp search --query "API documentation" \
  --content-filter --document-types "api_reference" \
  --metadata-filter --conditions "language=python" \
  --compose-filters "OR" --fallback-enabled

# Complex enterprise search with weighted filters
mcp search --query "security implementation" \
  --temporal-filter --freshness-weight 0.3 \
  --content-filter --semantic-weight 0.5 \
  --metadata-filter --metadata-weight 0.7 \
  --combine-weighted
```

### Personalized and Ranked Results

**Scenario**: Customizing results based on user preferences and context

```bash
# Personalized search based on user profile
mcp search --query "database optimization" --personalized \
  --user-profile "experience=intermediate,languages=python,role=backend" \
  --learning-enabled

# Context-aware ranking with user preferences
mcp search --query "testing strategies" --ranked \
  --context "framework=django,project_size=large" \
  --rank-strategy "hybrid" --preference-learning

# Collaborative filtering based search
mcp search --query "deployment tools" --personalized \
  --similar-users-weight 0.4 --content-based-weight 0.6
```

## 🎯 Specialized Use Cases

### Legal and Compliance Research

**Scenario**: Staying updated with regulatory changes

**Monitoring Setup**:

```bash
# Track regulatory websites
mcp scrape --url "https://regulatory-body.gov/updates" --monitor-changes
mcp scrape --url "https://legal-updates.com" --save-timeline

# Search for specific compliance topics
mcp search --query "data privacy requirements GDPR"
mcp search --query "security compliance standards"
```

### Educational Content Creation

**Scenario**: Developing training materials and courses

**Research and Compilation**:

```bash
# Gather educational resources
mcp scrape --url "https://educational-site.edu/courses" --extract-curriculum
mcp scrape --url "https://tutorial-site.com" --save-examples

# Search for pedagogical content
mcp search --query "learning objectives best practices"
mcp search --query "practical exercises hands-on examples"
```

### Product Documentation Maintenance

**Scenario**: Keeping internal docs current with external changes

**Synchronization Workflow**:

```bash
# Monitor external dependencies
mcp scrape --url "https://dependency.com/changelog" --track-versions
mcp scrape --url "https://api-provider.com/docs" --monitor-changes

# Update internal documentation
mcp search --query "version compatibility requirements"
mcp search --query "migration guide breaking changes"
```

## 🔧 Troubleshooting Recipes

### When Search Results Are Poor

**Diagnosis Steps**:

```bash
# Test with known-good queries
mcp search --query "system overview" --debug
mcp search --query "getting started guide" --verbose

# Check index status
mcp list-collections --status
mcp collection-info --name "main" --details
```

**Improvement Strategies**:

- Rephrase queries using different terminology
- Break complex questions into simpler parts
- Use more specific context in queries
- Try alternative search approaches

### When Scraping Fails

**Diagnostic Commands**:

```bash
# Test with simple sites first
mcp scrape --url "https://example.com" --test-mode
mcp scrape --url "https://httpbin.org/html" --verbose

# Check tier selection
mcp scrape --url "https://target-site.com" --show-tier
```

**Common Solutions**:

- Allow more time for complex sites
- Try different URLs within the same site
- Check if site requires special handling
- Verify site accessibility and availability

## 📈 Performance Optimization Recipes

### Enhanced Database Performance

**New in this version**: The system includes intelligent database connection pooling that automatically optimizes performance:

- **50.9% faster search response times** across all query types
- **887.9% higher throughput** during peak usage periods
- **Automatic scaling** that adjusts to your usage patterns
- **Smart resource management** with no configuration required

### Speeding Up Searches

**For Even Faster Results**:

```bash
# Use specific collections (now even faster with connection pooling)
mcp search --query "your terms" --collection "small-collection"

# Limit result count (performance improvements are most noticeable here)
mcp search --query "your terms" --limit 5

# Use simpler queries for exploration (benefits from improved throughput)
mcp search --query "overview guide" --quick
```

### Taking Advantage of Performance Improvements

**High-Throughput Workflows** (new capabilities):

```bash
# Batch multiple searches - system now handles concurrent requests much better
mcp search --query "configuration setup" &
mcp search --query "troubleshooting guide" &
mcp search --query "best practices" &
wait

# Complex federated analysis workflows that benefit from enhanced performance
mcp search --query "detailed technical analysis" --federated \
  --collections "docs,examples,tutorials" --use-hyde --limit 20

# Parallel federated searches across different collection groups
mcp search --query "backend architecture" --federated \
  --collections "api-docs,infrastructure" --mode "parallel" &
mcp search --query "frontend patterns" --federated \
  --collections "ui-docs,examples" --mode "parallel" &
wait
```

### Efficient Scraping Workflows

**Batch Processing**:

```bash
# Group similar requests
mcp scrape --urls-file "documentation-sites.txt" --batch-mode

# Use appropriate delays
mcp scrape --url-list "sites.txt" --delay 2s --respectful
```

## 🎓 Learning and Development

### Getting Started Workflow

**For New Users**:

1. Start with the [Quick Start Guide](./quick-start.md)
2. Try simple searches on familiar topics
3. Experiment with web scraping on simple sites
4. Gradually work up to more complex use cases
5. Reference these examples when you need specific patterns

### Building Expertise

**Progressive Learning**:

1. Master basic search techniques
2. Learn to identify good vs. poor search queries
3. Understand when different scraping tiers are triggered
4. Develop intuition for troubleshooting issues
5. Create your own recipe variations

### Advanced Techniques

**When You're Ready**:

- Combine multiple data sources effectively
- Develop systematic research workflows
- Create monitoring and alerting systems
- Build comprehensive knowledge management systems

## 🔗 Related Resources

- **[Quick Start Guide](./quick-start.md)**: Get up and running quickly
- **[Search & Retrieval](./search-and-retrieval.md)**: Master search techniques
- **[Web Scraping](./web-scraping.md)**: Understand the scraping system
- **[Troubleshooting](./troubleshooting.md)**: Solve common problems
- **Developer Resources**: See [../developers/index.md](../developers/index.md) for API integration

---

_📚 These recipes provide proven patterns for real-world success. Adapt them to your specific needs and build your own variations!_
