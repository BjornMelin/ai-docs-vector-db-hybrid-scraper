# Developer Getting Started Guide

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: Complete setup guide for developers  
> **Audience**: New developers joining the project

Get your development environment set up and make your first contribution! This guide takes you from zero to productive development in about 30 minutes.

## 🚀 Quick Start (5 minutes)

### Prerequisites Check

```bash
# Required tools - install if missing
python --version    # Need 3.13+
uv --version       # Need latest uv package manager
docker --version   # Need Docker Desktop
git --version      # Need Git
```

### Fast Setup

```bash
# Clone and setup
git clone <repository-url>
cd ai-docs-vector-db
uv sync

# Start services and test
./scripts/start-services.sh
uv run pytest tests/unit/ --tb=short -q
```

If everything passes, you're ready to develop! Skip to [First Contribution](#-first-contribution).

## 📋 Detailed Setup

### 1. System Prerequisites

#### Python 3.13+

```bash
# Check version
python --version

# Install if needed (Linux/macOS)
curl -sSL https://install.python.org/python-3.13.sh | bash

# Windows: Download from python.org
```

#### UV Package Manager

```bash
# Install uv (10-100x faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

#### Docker Desktop

- **Linux**: `sudo apt install docker.io` or use Docker Engine
- **macOS/Windows**: Download Docker Desktop
- **Verify**: `docker run hello-world`

#### Git Configuration

```bash
# Set up Git identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Enable helpful Git settings
git config --global init.defaultBranch main
git config --global pull.rebase false
```

### 2. Repository Setup

#### Clone Repository

```bash
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper
```

#### Environment Setup

```bash
# Install all dependencies
uv sync

# Activate virtual environment (automatic with uv commands)
source .venv/bin/activate  # Optional: for direct python usage

# Install pre-commit hooks
uv run pre-commit install
```

#### Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys (required for full functionality)
nano .env  # or your preferred editor
```

**Required environment variables:**

```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
# Other keys are optional for basic development
```

### 3. Service Dependencies

#### Start Core Services

```bash
# Start Qdrant (vector database) and DragonflyDB (cache)
./scripts/start-services.sh

# Verify services are running
curl localhost:6333/health        # Qdrant health check
curl localhost:6379/ping          # DragonflyDB ping
```

#### Service Configuration

The system uses Docker Compose for service orchestration:

- **Qdrant**: Vector database on port 6333
- **DragonflyDB**: Redis-compatible cache on port 6379
- **Optional**: Additional services for specific features

### 4. Development Environment

#### VS Code Setup (Recommended)

```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff

# Open project
code .
```

**VS Code settings (automatically configured):**

- Python interpreter: `.venv/bin/python`
- Linter: Ruff (fast Python linter)
- Formatter: Ruff (fast Python formatter)
- Type checker: Mypy integration

#### IDE Configuration

```json
// .vscode/settings.json (created automatically)
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "editor.formatOnSave": true
}
```

### 5. Verify Installation

#### Run Test Suite

```bash
# Quick test run (should complete in <30 seconds)
uv run pytest tests/unit/ --tb=short -q

# Full test suite with coverage
uv run pytest --cov=src --cov-report=term-missing

# Expected output: 500+ tests, 90%+ coverage
```

#### Code Quality Checks

```bash
# Linting and formatting (should pass without errors)
ruff check . --fix
ruff format .

# Type checking
uv run mypy src/

# All quality checks
make check  # or run individually
```

#### API Health Check

```bash
# Start the MCP server (in another terminal)
uv run python src/unified_mcp_server.py

# Test API endpoints
curl -X POST localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test search", "limit": 5}'
```

## 🛠️ Development Workflow

### 1. Creating a Feature Branch

```bash
# Always start from main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Start development
```

### 2. Development Loop

```bash
# 1. Write tests first (TDD approach)
uv run pytest tests/unit/test_your_feature.py -v

# 2. Implement feature
# Edit src/ files

# 3. Run tests to verify
uv run pytest tests/unit/test_your_feature.py -v

# 4. Run quality checks
ruff check . --fix && ruff format .
uv run mypy src/

# 5. Commit when ready
git add .
git commit -m "feat(module): implement new feature"
```

### 3. Pre-Commit Validation

The project automatically runs quality checks before commits:

```bash
# Automatic checks on git commit:
# 1. Ruff linting and formatting
# 2. Fast test suite
# 3. Import sorting
# 4. Basic type checking

# Manual validation
make check
```

## 📖 Understanding the Codebase

### Project Structure

```bash
src/
├── config/          # Configuration management
│   ├── models.py    # Pydantic configuration models
│   ├── loader.py    # Environment loading
│   └── validators.py # Config validation
├── models/          # API data models
│   ├── api_contracts.py    # Request/response models
│   ├── document_processing.py # Document models
│   └── vector_search.py    # Search models
├── services/        # Business logic layer
│   ├── embeddings/  # Embedding generation
│   ├── vector_db/   # Database operations
│   ├── crawling/    # Web scraping
│   └── cache/       # Caching layer
├── mcp_tools/       # MCP protocol tools
├── infrastructure/ # Client management
└── utils/          # Shared utilities
```

### Key Technologies

- **FastAPI**: Modern async web framework
- **Pydantic v2**: Data validation and serialization
- **Qdrant**: Vector similarity search database
- **AsyncIO**: Asynchronous programming
- **UV**: Fast Python package management
- **Ruff**: Fast Python linting and formatting

### Coding Standards

```python
# Example code style
from __future__ import annotations

import asyncio
from typing import Optional

from pydantic import BaseModel, Field
from src.models.api_contracts import SearchRequest


class ExampleService:
    """Example service following project patterns."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def process_request(
        self,
        request: SearchRequest,
        *,
        timeout: float = 30.0
    ) -> SearchResponse:
        """Process search request with proper typing and docs.
        
        Args:
            request: Validated search request
            timeout: Request timeout in seconds
            
        Returns:
            Search response with results
            
        Raises:
            ValidationError: If request is invalid
            ServiceError: If processing fails
        """
        # Implementation here
        pass
```

## 🧪 Testing Philosophy

### Test-Driven Development (TDD)

```bash
# 1. Write a failing test
uv run pytest tests/unit/test_feature.py::test_new_functionality -v
# Expected: FAILED (test doesn't exist yet)

# 2. Write minimal code to pass
# Add just enough implementation

# 3. Verify test passes
uv run pytest tests/unit/test_feature.py::test_new_functionality -v
# Expected: PASSED

# 4. Refactor and add more tests
# Add edge cases, error scenarios
```

### Testing Best Practices

1. **Comprehensive Coverage**: Test all public APIs
2. **Edge Cases**: Test boundary conditions and errors
3. **Fast Execution**: Unit tests should run in milliseconds
4. **Clear Assertions**: Make test failures informative
5. **Isolated Tests**: No shared state between tests

### Test Categories

```bash
# Model validation tests (Pydantic)
uv run pytest tests/unit/models/ -v

# Configuration tests
uv run pytest tests/unit/config/ -v

# Service integration tests
uv run pytest tests/unit/services/ -v

# Security validation tests
uv run pytest tests/unit/test_security.py -v
```

## 🔧 Common Development Tasks

### Adding a New API Endpoint

1. **Define models** in `src/models/api_contracts.py`
2. **Write tests** for request/response validation
3. **Implement logic** in appropriate service
4. **Add endpoint** to MCP tools
5. **Test integration** with full request cycle

### Adding a New Service

1. **Create service class** inheriting from `BaseService`
2. **Write comprehensive tests** for all methods
3. **Add configuration** models if needed
4. **Register service** in dependency injection
5. **Document** public APIs

### Debugging Issues

```bash
# Run specific test with detailed output
uv run pytest tests/unit/path/to/test.py::test_name -vvv

# Debug with Python debugger
uv run python -m pytest --pdb tests/unit/test_file.py

# Check logs
tail -f logs/application.log

# Profile performance
uv run pytest --profile tests/unit/
```

## 📚 First Contribution

### 1. Pick an Issue

- Look for "good first issue" labels
- Check existing GitHub issues
- Ask in discussions for guidance

### 2. Development Process

```bash
# Create branch
git checkout -b fix/issue-description

# Make changes following TDD
# 1. Write tests
# 2. Implement feature
# 3. Verify tests pass

# Quality check
make check

# Commit and push
git commit -m "fix(module): resolve issue description"
git push origin fix/issue-description
```

### 3. Pull Request

1. **Create PR** with descriptive title
2. **Reference issue** number if applicable
3. **Include tests** for all changes
4. **Update documentation** if needed
5. **Respond to feedback** promptly

### 4. PR Requirements

- [ ] All tests pass
- [ ] Code follows linting standards
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] No secrets or credentials committed
- [ ] Commit messages follow conventional format

## 🔗 Development Resources

### Documentation

- **[Architecture Guide](./architecture.md)**: System design and components
- **[API Reference](./api-reference.md)**: Complete API documentation
- **[Contributing Guide](./contributing.md)**: Detailed contribution guidelines

### External Resources

- **[FastAPI Documentation](https://fastapi.tiangolo.com/)**: Web framework
- **[Pydantic Documentation](https://docs.pydantic.dev/)**: Data validation
- **[Qdrant Documentation](https://qdrant.tech/documentation/)**: Vector database
- **[UV Documentation](https://docs.astral.sh/uv/)**: Package manager

### Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community help
- **Pull Requests**: Code review and collaboration

## 🆘 Troubleshooting

### Common Setup Issues

#### "uv: command not found"

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

#### "Docker daemon not running"

```bash
# Start Docker service
sudo systemctl start docker  # Linux
# Or start Docker Desktop application
```

#### "pytest: command not found"

```bash
# Use uv to run pytest
uv run pytest
# Instead of just: pytest
```

#### Test failures due to missing services

```bash
# Ensure services are running
./scripts/start-services.sh

# Check service status
docker ps
curl localhost:6333/health
```

### Getting Help

1. **Check existing issues** on GitHub
2. **Search documentation** for answers
3. **Ask in discussions** for community help
4. **Create detailed issue** with reproduction steps

## 🎯 Next Steps

### Immediate Next Steps

1. ✅ Complete setup verification
2. ✅ Run full test suite successfully
3. ✅ Make a small test change and commit
4. ✅ Read [Architecture Guide](./architecture.md)
5. ✅ Review [API Reference](./api-reference.md)

### Learning Path

1. **Week 1**: Understand project structure and make first PR
2. **Week 2**: Contribute to existing features and tests
3. **Week 3**: Design and implement new features
4. **Week 4**: Review and mentor other contributions

### Advanced Topics

- **Performance Optimization**: Profiling and optimization techniques
- **Security Practices**: Secure coding and vulnerability assessment
- **Deployment**: Docker, Kubernetes, and cloud deployment
- **Monitoring**: Logging, metrics, and observability

---

*🛠️ You're ready to start developing! The codebase is well-structured, thoroughly tested, and designed for easy contribution. Start with small changes to get familiar with the workflow, then tackle larger features as you build confidence.*
