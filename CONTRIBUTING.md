# 🤝 Contributing to Advanced AI Documentation Scraper

We welcome contributions to make this project even better! This guide will help you get started with
contributing to our research-backed advanced documentation scraping system.

## 🚀 Quick Start for Contributors

### Prerequisites

- **Python 3.13+** with `uv` package manager
- **Docker Desktop** for local development
- **Git** for version control
- **Node.js 18+** for MCP server testing

### Development Setup

1. **Fork and Clone**

   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-docs-vector-db-hybrid-scraper.git
   cd ai-docs-vector-db-hybrid-scraper
   ```

2. **Environment Setup**

   ```bash
   # Install uv if not already available
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Run setup script
   chmod +x setup.sh
   ./setup.sh

   # Create development environment
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start Development Services**

   ```bash
   # Start Qdrant database
   docker-compose up -d

   # Verify setup
   curl http://localhost:6333/health
   ```

## 🛠️ Development Guidelines

### Code Standards

We follow **modern best practices** for clean, maintainable code:

#### Python Code Style

```bash
# Format and lint all code before committing
ruff check . --fix
ruff format .

# Type checking
uv run mypy src/

# Run tests with coverage (CI profile)
uv run python scripts/dev.py test --profile ci
```

#### Required Code Quality

- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: Follow Google-style docstrings for all public functions
- **Testing**: Maintain >=90% test coverage (threshold enforced in CI; run `uv run python scripts/dev.py test --profile ci` locally when needed)
- **Performance**: Follow advanced performance patterns

#### Example Code Structure

```python
from typing import Any
from pydantic import BaseModel, Field


class ExampleConfig(BaseModel):
    """Example configuration following modern patterns.

    Args:
        param1: Description of parameter
        param2: Another parameter with default
    """

    param1: str = Field(..., description="Required parameter")
    param2: int = Field(default=100, description="Optional parameter")


async def example_function(config: ExampleConfig) -> dict[str, Any]:
    """Example function following modern conventions.

    Args:
        config: Configuration object

    Returns:
        Dictionary with processed results

    Raises:
        ValueError: If configuration is invalid
    """
    # Implementation here
    return {"status": "success"}
```

### Git Workflow

#### Branch Naming

- `feat/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/update-topic` - Documentation updates
- `perf/optimization-area` - Performance improvements
- `test/test-area` - Test additions/improvements

#### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Good commit messages
feat: add BGE-reranker-v2-m3 integration for 10-20% accuracy improvement
fix: resolve memory leak in FastEmbed model initialization
docs: update MCP server configuration for Claude Desktop
perf: optimize vector quantization for 50% storage reduction
test: add integration tests for hybrid search functionality

# Bad commit messages (avoid these)
fix: bug
update: stuff
change: code
```

#### Pull Request Process

1. **Create Feature Branch**

   ```bash
   git checkout -b feat/your-feature-name
   git push -u origin feat/your-feature-name
   ```

2. **Make Changes**

   - Follow coding standards
   - Add/update tests
   - Update documentation
   - Test locally

3. **Pre-commit Checklist**

   ```bash
   # Run full test suite
   uv run python scripts/dev.py test --profile ci

   # Code quality checks
   ruff check . --fix
   ruff format .

   # Type checking
   uv run mypy src/

   # Documentation build test
   uv run mkdocs build -f docs/build_config/mkdocs.yml
   ```

4. **Submit Pull Request**
   - Use descriptive title and description
   - Reference related issues
   - Include testing information
   - Add screenshots for UI changes

### GitHub Actions Workflows

- The **Core CI** workflow (`core-ci.yml`) runs automatically on every pull
  request and must pass before a merge.
- The **Documentation Checks** workflow triggers when Markdown or docs assets
  change.

Two additional workflows are available on demand when you need deeper
validation:

| Workflow                                                  | How to run                                                                                                                                                                                                                                        | Inputs                                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Security Scan (On-Demand)**<br>`security-opt-in.yml`    | 1. Open the **Actions** tab → **Security Scan (On-Demand)**.<br>2. Click **Run workflow** and pick your branch.<br>3. Toggle the optional inputs as needed.<br>4. Review the uploaded artifacts (`dependency-security-reports`, `bandit-report`). | `include_bandit` (default `true`)<br>`include_safety` (default `true`)  |
| **Extended Tests (On-Demand)**<br>`regression-opt-in.yml` | 1. Navigate to **Actions** → **Extended Tests (On-Demand)**.<br>2. Choose your branch and press **Run workflow**.<br>3. Enable benchmark execution if required.<br>4. Inspect artifacts (`regression-coverage`, `benchmark-results`).             | `run_full_tests` (default `true`)<br>`run_benchmarks` (default `false`) |

Both workflows also trigger automatically on pull requests when their path
filters match (e.g., dependency manifest or test suite changes). Use them before
merging riskier updates to keep the default CI experience fast while still
gaining full security and regression coverage when it matters.

## 🧪 Testing Guidelines

### Test Structure

```plaintext
tests/
├── unit/                 # Unit tests for individual components
├── services/             # Narrow integration shims
├── contracts/            # API/DTO schema checks
├── data_quality/         # Dataset validators
└── fixtures/            # Test data and fixtures
```

### Writing Tests

#### Unit Tests

```python
import pytest
from src.crawl4ai_bulk_embedder import EmbeddingConfig, EmbeddingProvider


@pytest.mark.asyncio
async def test_embedding_config_validation():
    """Test advanced embedding configuration validation."""
    config = EmbeddingConfig(
        provider=EmbeddingProvider.HYBRID,
        enable_reranking=True,
        rerank_top_k=20
    )

    assert config.provider == EmbeddingProvider.HYBRID
    assert config.enable_reranking is True
    assert config.rerank_top_k == 20
```

#### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_scraping_pipeline():
    """Test complete advanced scraping and embedding pipeline."""
    # Test implementation
    pass
```

#### Performance Tests

```python
@pytest.mark.benchmark
def test_embedding_generation_speed(benchmark):
    """Benchmark embedding generation performance."""
    def create_embedding():
        # Implementation
        pass

    result = benchmark(create_embedding)
    assert result.duration < 0.1  # <100ms target
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m benchmark

# Run with coverage
uv run python scripts/dev.py test --profile ci

# Run performance benchmarks
uv run pytest --benchmark-only
```

## 📚 Documentation Contributions

### Documentation Types

- **API Documentation**: Auto-generated from docstrings
- **User Guides**: Step-by-step tutorials in `/docs`
- **Technical Specs**: Implementation details and research
- **Examples**: Working code examples in `/examples`

### Writing Documentation

#### User Guides

- Use clear, actionable headings
- Include code examples for every concept
- Test all code examples
- Use consistent formatting

#### API Documentation

- Complete docstrings for all public functions
- Include usage examples
- Document all parameters and return values
- Note any side effects or requirements

### Building Documentation

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Serve documentation locally
uv run mkdocs serve -f docs/build_config/mkdocs.yml

# Build documentation
uv run mkdocs build -f docs/build_config/mkdocs.yml
```

## 🎯 Contribution Areas

### High-Priority Areas

#### 🔬 Research & Optimization

- **Embedding Models**: Test and integrate new high-performance models
- **Vector Quantization**: Explore advanced quantization techniques
- **Hybrid Search**: Improve sparse-dense fusion algorithms
- **Reranking**: Experiment with new cross-encoder models
- **Database Optimization Research**: Contribute to ML-based connection pool optimization (see [Database Optimization Research](#database-optimization-research))

#### ⚡ Performance Improvements

- **Memory Optimization**: Reduce memory footprint
- **Parallel Processing**: Improve concurrent operations
- **Caching**: Implement intelligent caching strategies
- **Batch Processing**: Optimize batch sizes and operations
- **Database Connection Pooling**: Improve connection management and scaling algorithms

#### 🤖 MCP Integration

- **New MCP Servers**: Add support for additional MCP servers
- **Claude Integration**: Improve Claude Desktop workflows
- **Real-time Updates**: Implement live documentation updates
- **Advanced Queries**: Support complex search patterns

#### 🌐 Crawler Enhancements

- **Site Support**: Add support for complex documentation sites
- **Content Extraction**: Improve content quality and structure
- **Error Handling**: Robust handling of edge cases
- **Monitoring**: Better observability and metrics

### Medium-Priority Areas

#### 📊 Analytics & Monitoring

- Performance dashboards
- Usage analytics
- Cost tracking
- Quality metrics

#### 🔧 DevOps & Infrastructure

- CI/CD improvements
- Docker optimizations
- Deployment automation
- Monitoring setup

#### 🎨 User Experience

- CLI improvements
- Configuration management
- Error messages
- Progress indicators

## 🐛 Bug Reports

### Before Reporting

1. **Search Existing Issues**: Check if the bug is already reported
2. **Reproduce Locally**: Ensure you can consistently reproduce the issue
3. **Check Documentation**: Verify it's not a configuration issue

### Bug Report Template

```markdown
## Bug Description

Clear and concise description of the bug.

## Environment

- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 14]
- Python Version: [e.g., 3.13.1]
- Package Versions: [output of `uv pip list`]
- Docker Version: [if applicable]

## Steps to Reproduce

1. Step 1
2. Step 2
3. Step 3

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Code/Configuration

Relevant code snippets or configuration files.

## Error Messages

Complete error messages and stack traces.

## Additional Context

Any other relevant information.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description

Clear description of the proposed feature.

## Problem Statement

What problem does this solve?

## Proposed Solution

How would you like this feature to work?

## Alternatives Considered

Other approaches you've considered.

## Research References

Links to papers, benchmarks, or implementations.

## Implementation Notes

Technical considerations or constraints.
```

## 🏆 Recognition

### Contributors

All contributors are recognized in our:

- **README.md**: Major contributors section
- **Release Notes**: Feature and fix acknowledgments
- **Documentation**: Author attribution
- **Social Media**: Community highlights

### Types of Contributions

- **Code**: New features, bug fixes, optimizations
- **Documentation**: Guides, examples, API docs
- **Research**: Performance analysis, model evaluation
- **Testing**: Test coverage, performance benchmarks
- **Community**: Issue triage, user support
- **Design**: UX/UI improvements, architecture

## 📞 Getting Help

### Community Support

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Discord/Slack**: Real-time community chat (link in README)

### Maintainer Contact

- **Email**: [maintainer email if available]
- **GitHub**: @BjornMelin
- **Response Time**: We aim to respond within 48 hours

### Office Hours

- **Weekly Community Call**: [If applicable]
- **Maintainer Office Hours**: [If applicable]

## 📝 Legal

### Contributor License Agreement

By contributing to this project, you agree that your contributions will be licensed under
the same [MIT License](LICENSE) as the project.

### Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone.
Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

### Attribution

All contributions are valued and will be appropriately attributed in project documentation and release notes.

---

## 🚀 Ready to Contribute?

1. **Start Small**: Look for "good first issue" labels
2. **Ask Questions**: Don't hesitate to ask for clarification
3. **Share Ideas**: Propose improvements and optimizations
4. **Test Thoroughly**: Ensure your changes work reliably
5. **Document**: Help others understand your contributions

---

## 🗄️ Database Optimization Research

### Performance Research Achievements

Our community has achieved remarkable database optimization results:

- **50.9% latency reduction** for complex queries
- **887.9% throughput increase** for concurrent operations
- **Dynamic connection scaling** based on ML-driven load prediction
- **Adaptive pool sizing** with real-time performance monitoring

### Contributing to Database Research

#### Research Areas

##### Connection Pool Optimization

- Machine learning models for predicting optimal pool sizes
- Adaptive scaling algorithms based on workload patterns
- Performance monitoring and automatic tuning
- Load balancing strategies for distributed deployments

##### Query Performance Analysis

- Query pattern recognition and optimization
- Caching strategies for frequently accessed data
- Index optimization recommendations
- Memory usage pattern analysis

##### Benchmarking and Metrics

- Performance regression detection
- Comparative analysis across different configurations
- Real-world workload simulation
- Cost-performance optimization

#### How to Contribute Research

1. **Performance Improvements**

   ```bash
   # Run baseline evaluation harness
   uv run python scripts/eval/rag_golden_eval.py \
       --dataset tests/data/rag/golden_set.jsonl \
       --output artifacts/rag_golden_report.json

   # Export focused metrics snapshot
   uv run python scripts/eval/rag_golden_eval.py \
       --dataset tests/data/rag/golden_set.jsonl \
       --metrics-allowlist config/metrics_allowlist.json \
       --output artifacts/rag_metrics_snapshot.json

   # Profile specific components
   uv run python -m cProfile -o profile.stats your_optimization.py
   ```

2. **ML Model Enhancements**

   - Contribute new prediction models for connection scaling
   - Improve feature engineering for workload classification
   - Experiment with different optimization algorithms
   - Submit performance analysis and comparisons

3. **Benchmarking Contributions**

   ```bash
   # Run comprehensive benchmarks
   uv run python scripts/eval/rag_golden_eval.py \
       --dataset tests/data/rag/golden_set.jsonl \
       --output artifacts/rag_golden_report.json

   # Generate performance reports
   uv run python scripts/benchmark_lightweight_tier.py --output-report
   ```

4. **Documentation of Findings**
   - Share benchmark results and analysis
   - Document optimization techniques and their impact
   - Contribute to research papers and technical reports
   - Create tutorials for implementing optimizations

#### Research Collaboration Guidelines

##### Sharing Results

- Include complete benchmark data and methodology
- Provide reproducible test cases
- Document hardware/software configurations
- Share raw performance metrics alongside analysis

##### Proposing Optimizations

- Start with performance analysis of current state
- Clearly describe the optimization approach
- Provide evidence of improvement (benchmarks/profiling)
- Consider edge cases and failure scenarios

##### Code Contributions

- Follow TDD approach for optimization features
- Include comprehensive performance tests
- Document performance characteristics in docstrings
- Provide configuration examples and usage guides

#### Research Issue Labels

When contributing research-related issues, use these labels:

- `research`: General research contributions
- `performance`: Performance improvement proposals
- `database-optimization`: Database-specific optimizations
- `ml-enhancement`: Machine learning model improvements
- `benchmarking`: Benchmark results and analysis

#### Getting Started with Research

1. **Set up benchmarking environment**

   ```bash
   # Install additional research dependencies
   uv add pytest-benchmark memory-profiler line-profiler

   # Run baseline evaluation
   uv run python scripts/eval/rag_golden_eval.py \
       --dataset tests/data/rag/golden_set.jsonl \
       --output artifacts/rag_golden_report.json
   ```

2. **Explore current optimizations**

   - Review `src/infrastructure/database/connection_manager.py`
   - Study the ML models in `src/infrastructure/database/load_monitor.py`
   - Review the evaluation harness under `scripts/eval/` and docs in `docs/testing/evaluation-harness.md`

3. **Join research discussions**
   - Comment on research-related GitHub issues
   - Share your performance findings and analyses
   - Propose new optimization approaches

---

## Thank you for helping make this project better! 🎉

Every contribution, no matter how small, helps advance the performance and capabilities of AI documentation
processing. Join our research community to push the boundaries of database optimization and ML-driven performance enhancements!
