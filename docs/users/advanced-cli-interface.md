# Advanced CLI Interface

The Advanced CLI Interface provides a rich, interactive command-line experience with comprehensive configuration management, batch operations, and real-time monitoring capabilities.

## Overview

The CLI interface offers:

- **Rich Console**: Beautiful, color-coded output with progress indicators
- **Configuration Wizard**: Interactive setup and validation
- **Auto-completion**: Intelligent command and parameter completion
- **Batch Operations**: Process multiple operations efficiently
- **Worker Management**: Monitor and control background workers
- **Real-time Monitoring**: Live system metrics and performance data

## Installation and Setup

### Initial Configuration

```bash
# Initialize the CLI with configuration wizard
uv run python -m src.cli.main init --interactive

# Or use quick setup with defaults
uv run python -m src.cli.main init --quick
```

### Configuration Wizard

The interactive configuration wizard guides you through:

1. **Project Settings**: Name, description, quality tier
2. **Provider Configuration**: Embedding and crawl providers
3. **Performance Tuning**: Thread counts, rate limits, timeouts
4. **Quality Preferences**: Accuracy vs. speed trade-offs
5. **Monitoring Setup**: Metrics and alerting preferences

```bash
# Run configuration wizard
$ cli-tool config wizard

✓ Project Configuration
  Project Name: My Documentation Project
  Quality Tier: balanced [economy/balanced/premium]
  Description: AI-powered documentation search

✓ Provider Settings  
  Embedding Provider: fastembed [openai/fastembed]
  Crawl Provider: crawl4ai [crawl4ai/firecrawl]
  Browser Automation: 5-tier [enabled/disabled]

✓ Performance Settings
  Max Workers: 4 [1-16]
  Rate Limit: 10 req/sec [1-100]
  Cache Size: 1000 items [100-10000]

Configuration saved successfully! ✨
```

## Core Commands

### Project Management

```bash
# Create new project with URLs
cli-tool project create "API Docs" \
  --urls https://api.example.com/docs \
  --quality-tier premium \
  --description "Complete API documentation"

# List all projects
cli-tool project list

# Get project details
cli-tool project info proj_123

# Update project settings
cli-tool project update proj_123 \
  --name "Updated API Docs" \
  --description "Enhanced documentation"

# Delete project
cli-tool project delete proj_123 --confirm
```

### Content Processing

```bash
# Add URLs to existing project
cli-tool content add proj_123 \
  --urls https://docs.example.com/guide \
  --urls https://docs.example.com/tutorial \
  --batch-size 5

# Bulk import from file
cli-tool content import proj_123 \
  --file urls.txt \
  --format txt \
  --parallel 3

# Process with content intelligence
cli-tool content analyze proj_123 \
  --enable-classification \
  --enable-quality-assessment \
  --confidence-threshold 0.7
```

### Search and Retrieval

```bash
# Interactive search
cli-tool search proj_123 \
  --query "how to authenticate users" \
  --limit 10 \
  --strategy hybrid

# Advanced search with filters
cli-tool search proj_123 \
  --query "python examples" \
  --content-type code \
  --min-quality 0.8 \
  --date-range "2024-01-01,2024-12-31"

# Export search results
cli-tool search proj_123 \
  --query "authentication guide" \
  --export results.json \
  --format json
```

### System Monitoring

```bash
# Real-time system status
cli-tool monitor status --live

# Performance metrics
cli-tool monitor metrics --detailed

# Worker management
cli-tool worker list
cli-tool worker start --count 2
cli-tool worker stop worker_123
cli-tool worker restart --all
```

## Batch Operations

### Bulk URL Processing

```bash
# Process multiple URLs with progress tracking
cli-tool batch process \
  --file urls.txt \
  --project proj_123 \
  --workers 4 \
  --progress \
  --retry-failed 3

# Example urls.txt format:
# https://docs.example.com/guide1
# https://docs.example.com/guide2  
# https://api.example.com/reference
```

### Configuration Templates

```bash
# Apply configuration template
cli-tool config apply \
  --template production \
  --project proj_123

# Save current config as template
cli-tool config save-template \
  --name my-setup \
  --include-secrets false

# List available templates
cli-tool config templates list
```

### Batch Quality Assessment

```bash
# Analyze content quality across projects
cli-tool quality batch-assess \
  --projects proj_123,proj_456 \
  --threshold 0.6 \
  --export quality-report.json \
  --include-suggestions
```

## Rich Console Features

### Progress Indicators

```bash
# Processing with rich progress display
$ cli-tool content add proj_123 --urls-file large-list.txt

Processing URLs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
 • Crawled: 245/245 URLs
 • Processed: 243/245 documents  
 • Errors: 2 URLs (see errors.log)
 • Quality: Avg 0.87, Min 0.45, Max 0.98
 • Cache Hits: 67/245 (27.3%)

✓ Processing completed in 2m 15s
```

### Interactive Selection

```bash
# Interactive project selection
$ cli-tool search --interactive

? Select project:
❯ API Documentation (proj_123) - 245 docs
  User Guides (proj_456) - 89 docs  
  Tutorials (proj_789) - 156 docs

? Search query: authentication examples

? Search strategy:
❯ Hybrid (recommended)
  Semantic
  Keyword
  Reranked
```

### Status Dashboard

```bash
# Live system dashboard
$ cli-tool monitor dashboard

┏━━━━━━━━━━━━━━━ System Status ━━━━━━━━━━━━━━━┓
┃ CPU Usage:     ████████░░ 80%          ┃
┃ Memory:        ████████░░ 75%          ┃  
┃ Disk I/O:      ███░░░░░░░ 30%          ┃
┃ Network:       ██████░░░░ 60%          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━ Active Workers ━━━━━━━━━━━━━━━┓
┃ Crawlers:      4/4 running              ┃
┃ Processors:    2/2 running              ┃
┃ Queue Size:    23 pending               ┃
┃ Processed:     1,247 today              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━ Performance ━━━━━━━━━━━━━━━━┓
┃ Avg Response:  125ms                    ┃
┃ Cache Hit:     78.5%                    ┃
┃ Success Rate:  97.8%                    ┃
┃ Uptime:        2d 14h 32m               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Auto-completion

### Bash Completion

```bash
# Install bash completion
cli-tool completion bash > ~/.cli-tool-completion
echo "source ~/.cli-tool-completion" >> ~/.bashrc

# Usage with tab completion
$ cli-tool project <TAB>
create  delete  info  list  update

$ cli-tool search proj_<TAB>
proj_123  proj_456  proj_789
```

### Zsh Completion

```bash
# Install zsh completion  
cli-tool completion zsh > ~/.zsh-completions/_cli-tool
echo "fpath=(~/.zsh-completions $fpath)" >> ~/.zshrc
```

## Configuration Management

### Environment-Specific Configs

```bash
# Set configuration for different environments
cli-tool config set --env development \
  --key crawl.rate_limit \
  --value 5

cli-tool config set --env production \
  --key crawl.rate_limit \
  --value 20

# Switch between environments
cli-tool config use development
cli-tool config use production
```

### Validation and Testing

```bash
# Validate configuration
cli-tool config validate

# Test configuration with sample operation
cli-tool config test \
  --operation crawl \
  --url https://example.com \
  --timeout 30s
```

### Migration Tools

```bash
# Migrate configuration from older versions
cli-tool config migrate \
  --from v1.0 \
  --to v2.0 \
  --backup config-backup.json

# Import/export configurations
cli-tool config export config.json
cli-tool config import config.json
```

## Worker Management

### Background Processing

```bash
# Start workers for background processing
cli-tool worker start \
  --type crawler \
  --count 3 \
  --queue-size 100

# Monitor worker performance
cli-tool worker monitor \
  --worker-id worker_123 \
  --metrics cpu,memory,queue

# Scale workers based on load
cli-tool worker autoscale \
  --min-workers 2 \
  --max-workers 8 \
  --cpu-threshold 80
```

### Queue Management

```bash
# View processing queue
cli-tool queue status

# Priority processing
cli-tool queue priority \
  --project proj_123 \
  --priority high

# Clear failed jobs
cli-tool queue clear-failed
```

## Integration Examples

### CI/CD Integration

```bash
# CI pipeline example
#!/bin/bash
set -e

# Validate configuration
cli-tool config validate

# Process documentation updates
cli-tool content add docs_project \
  --urls-file updated-docs.txt \
  --wait-completion \
  --fail-on-error

# Verify quality standards
cli-tool quality check docs_project \
  --min-score 0.7 \
  --fail-below-threshold

echo "Documentation processing completed successfully"
```

### Scheduled Operations

```bash
# Cron job for regular updates (add to crontab)
0 2 * * * /usr/local/bin/cli-tool content refresh-all --quality-check

# Daily quality reports
0 8 * * * /usr/local/bin/cli-tool quality report --email admin@company.com
```

## Troubleshooting

### Debug Mode

```bash
# Enable detailed debugging
cli-tool --debug search proj_123 --query "test"

# Save debug output to file
cli-tool --debug --log-file debug.log content add proj_123 --urls https://example.com
```

### Health Checks

```bash
# Comprehensive system health check
cli-tool health check --detailed

# Test specific components
cli-tool health check --component embedding
cli-tool health check --component qdrant
cli-tool health check --component crawling
```

### Performance Analysis

```bash
# Generate performance report
cli-tool performance analyze \
  --time-range "last 24h" \
  --export perf-report.html \
  --include-recommendations
```

For advanced usage and API integration, see the [Developer Integration Guide](../developers/integration-guide.md).