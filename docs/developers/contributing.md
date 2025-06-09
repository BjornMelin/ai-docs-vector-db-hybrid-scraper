# Contributing Guide

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: Complete contribution guidelines and workflow  
> **Audience**: Contributors and team members

Welcome to the AI Documentation Vector DB project! This guide covers everything you need to know about contributing code, documentation, and improvements to the project.

## 🚀 Quick Contributing Start

### Before You Start

1. ✅ Read the [Getting Started Guide](./getting-started.md) and set up your environment
2. ✅ Check existing [GitHub Issues](https://github.com/project/issues) for work opportunities
3. ✅ Join our discussions to connect with the community
4. ✅ Review this guide for detailed workflow instructions

### Fast Track Contribution

```bash
# 1. Fork and clone
git clone your-fork-url
cd ai-docs-vector-db

# 2. Create branch
git checkout -b feature/your-improvement

# 3. Make changes with tests
# Edit code, add tests, ensure quality

# 4. Submit PR
git push origin feature/your-improvement
# Create PR via GitHub interface
```

## 📋 Contribution Types

### 🐛 Bug Fixes

- **High Impact**: Critical bugs affecting core functionality
- **Medium Impact**: Bugs in specific features or edge cases  
- **Low Impact**: Minor issues like typos or cosmetic problems

### ✨ New Features

- **Core Features**: Search, embedding, document processing
- **API Enhancements**: New endpoints or improved responses
- **Integrations**: New data sources or external service support
- **Performance**: Speed, memory, or efficiency improvements

### 📚 Documentation

- **API Documentation**: Endpoint descriptions and examples
- **User Guides**: How-to guides and tutorials
- **Developer Docs**: Architecture and technical documentation
- **Examples**: Code samples and use case demonstrations

### 🧪 Testing

- **Unit Tests**: Individual component testing
- **Integration Tests**: Service interaction testing
- **Performance Tests**: Benchmarking and load testing
- **Security Tests**: Vulnerability and safety testing

### 🛠️ Infrastructure

- **CI/CD**: GitHub Actions and automation
- **Docker**: Container improvements
- **Dependencies**: Package updates and security fixes
- **Monitoring**: Logging, metrics, and observability

## 🔄 Development Workflow

### 1. Issue Selection and Planning

#### Finding Work

```bash
# Good starting points
- Look for "good first issue" labels
- Check "help wanted" issues
- Browse open feature requests
- Review documentation improvement needs
```

#### Issue Communication

1. **Comment on issues** you're interested in working on
2. **Ask questions** if requirements are unclear
3. **Propose approach** for complex features
4. **Update progress** regularly on long-running work

### 2. Branch Strategy

#### Branch Naming Convention

```bash
# Feature branches
feature/add-hybrid-search
feature/improve-chunking-algorithm
feature/api-rate-limiting

# Bug fixes
fix/memory-leak-embedding-cache
fix/validation-error-handling
fix/docker-networking-issue

# Documentation
docs/update-api-reference
docs/add-testing-guide
docs/improve-user-tutorials

# Refactoring
refactor/service-layer-architecture
refactor/unify-configuration-system
```

#### Branch Workflow

```bash
# Always start from main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Work on feature with regular commits
git add .
git commit -m "feat(module): implement part of feature"

# Keep branch updated
git fetch origin
git rebase origin/main  # Preferred over merge

# Push when ready for review
git push origin feature/your-feature-name
```

### 3. Commit Message Standards

#### Conventional Commits Format

```bash
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Commit Types

- **feat**: New feature for users
- **fix**: Bug fix for users
- **docs**: Documentation changes
- **test**: Adding or modifying tests
- **refactor**: Code changes that don't fix bugs or add features
- **perf**: Performance improvements
- **ci**: CI/CD pipeline changes
- **chore**: Maintenance tasks, dependency updates

#### Examples

```bash
# Feature additions
feat(search): add hybrid vector search with reranking
feat(api): implement rate limiting for embedding endpoints
feat(models): add Pydantic v2 validation for API contracts

# Bug fixes
fix(embedding): resolve memory leak in batch processing
fix(config): handle missing environment variables gracefully
fix(security): validate URLs against SSRF attacks

# Documentation
docs(api): update API reference with new models
docs(tutorial): add web scraping best practices guide
docs(architecture): document service layer patterns

# Tests
test(models): add comprehensive validation tests for SearchRequest
test(security): add URL validation edge case tests
test(performance): add benchmark tests for search latency

# Breaking changes
feat(config)!: migrate to unified configuration system

BREAKING CHANGE: Configuration structure has changed. 
Run migration script to update existing configs.
```

#### Commit Best Practices

1. **Use present tense**: "Add feature" not "Added feature"
2. **Be descriptive**: Explain what and why, not just what
3. **Keep first line under 50 characters**
4. **Use body for complex changes**: Explain motivation and impact
5. **Reference issues**: Include issue numbers when applicable

## 🧪 Testing Requirements

### Test-Driven Development (TDD)

#### TDD Process

```bash
# 1. Write failing test
uv run pytest tests/unit/test_new_feature.py::test_feature_success -v
# Expected: FAILED (no implementation yet)

# 2. Write minimal code to pass test
# Add just enough implementation to make test pass

# 3. Run test again
uv run pytest tests/unit/test_new_feature.py::test_feature_success -v
# Expected: PASSED

# 4. Refactor and add more tests
# Add edge cases, error scenarios, performance tests

# 5. Run full test suite
uv run pytest tests/unit/ --tb=short
```

### Test Coverage Requirements

#### Coverage Standards

- **Critical modules**: 95%+ required (models, security, core APIs)
- **Service layer**: 90%+ required
- **Utilities**: 85%+ acceptable
- **New features**: 90%+ required for all new code

#### Coverage Verification

```bash
# Generate coverage report
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Check coverage for specific modules
uv run pytest --cov=src/models --cov-report=term-missing

# Coverage must not decrease with new changes
```

### Testing Best Practices

#### Test Organization

```python
"""Unit tests for SearchService."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.services.search import SearchService
from src.models.api_contracts import SearchRequest, SearchResponse


class TestSearchService:
    """Test class for SearchService."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create mock vector database for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def search_service(self, mock_vector_db):
        """Create SearchService instance for testing."""
        return SearchService(vector_db=mock_vector_db)
    
    async def test_search_success_scenario(self, search_service):
        """Test successful search execution."""
        # Arrange
        request = SearchRequest(query="test query", limit=5)
        
        # Act
        response = await search_service.search(request)
        
        # Assert
        assert response.success is True
        assert len(response.results) <= 5
    
    async def test_search_validation_error(self, search_service):
        """Test search with invalid request."""
        # Arrange
        invalid_request = SearchRequest(query="", limit=0)
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await search_service.search(invalid_request)
        assert "query cannot be empty" in str(exc_info.value)
```

#### Pydantic Model Testing

```python
def test_search_request_validation():
    """Test SearchRequest field validation."""
    # Valid request
    request = SearchRequest(query="test", limit=10)
    assert request.query == "test"
    assert request.limit == 10
    
    # Test defaults
    request_minimal = SearchRequest(query="test")
    assert request_minimal.limit == 20  # Default value
    
    # Test constraints
    with pytest.raises(ValidationError):
        SearchRequest(query="", limit=10)  # Empty query
    
    with pytest.raises(ValidationError):
        SearchRequest(query="test", limit=0)  # Invalid limit
```

#### Async Service Testing

```python
@pytest.mark.asyncio
async def test_service_lifecycle():
    """Test service initialization and cleanup."""
    service = ServiceClass(config)
    
    # Test initialization
    await service.initialize()
    assert service.is_initialized is True
    
    try:
        # Test functionality
        result = await service.process_request(request)
        assert result.success is True
    finally:
        # Ensure cleanup
        await service.cleanup()
        assert service.is_initialized is False
```

## 🎯 Code Quality Standards

### Python Code Style

#### Core Standards

```python
# Use type hints throughout
def process_documents(
    documents: list[Document],
    *,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[Chunk]:
    """Process documents into chunks.
    
    Args:
        documents: List of documents to process
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of document chunks
        
    Raises:
        ValidationError: If documents are invalid
        ProcessingError: If chunking fails
    """

# Use Pydantic for data validation
class DocumentRequest(BaseModel):
    """Request to process documents."""
    
    url: HttpUrl = Field(..., description="Document URL to process")
    chunk_size: int = Field(1000, ge=100, le=5000)
    include_metadata: bool = Field(True)
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL is not a local address."""
        if any(host in v for host in ['localhost', '127.0.0.1', '192.168']):
            raise ValueError('Local URLs are not allowed')
        return v

# Use async/await properly
async def process_request(request: DocumentRequest) -> DocumentResponse:
    """Process document request asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(str(request.url)) as response:
            content = await response.text()
            return await process_content(content)
```

#### Naming Conventions

```python
# Files and modules: snake_case.py
search_service.py
document_processor.py
api_contracts.py

# Classes: PascalCase
class SearchService:
class DocumentProcessor:
class APIContract:

# Functions and methods: snake_case
def search_documents():
def process_content():
def validate_request():

# Constants: UPPER_SNAKE_CASE
MAX_CHUNK_SIZE = 5000
DEFAULT_TIMEOUT = 30.0
API_VERSION = "v1"

# Private members: _leading_underscore
class Service:
    def __init__(self):
        self._client = None
        self._config = {}
    
    def _internal_method(self):
        pass
```

### Code Quality Tools

#### Linting and Formatting

```bash
# Ruff (fast linting and formatting)
ruff check . --fix     # Fix linting issues
ruff format .          # Format code

# Type checking
uv run mypy src/       # Static type checking

# Import sorting
ruff check --select I --fix .  # Sort imports

# Security scanning
bandit -r src/         # Security vulnerability scanning
```

#### Pre-commit Validation

```bash
# Automatic quality checks before commit
pre-commit install

# Manual quality check
make check

# Full validation including tests
make validate
```

### Documentation Standards

#### Docstring Format (Google Style)

```python
def search_documents(
    query: str,
    collection_name: str = "documents",
    limit: int = 10,
    score_threshold: float = 0.7
) -> SearchResponse:
    """Search documents using vector similarity.
    
    Performs semantic search across the specified collection using
    hybrid vector search with configurable score thresholds.
    
    Args:
        query: Search query string
        collection_name: Target collection name
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score (0.0-1.0)
        
    Returns:
        SearchResponse containing matching documents with scores
        
    Raises:
        ValidationError: If query is empty or parameters are invalid
        ServiceError: If vector database is unavailable
        
    Example:
        >>> response = search_documents("machine learning", limit=5)
        >>> print(f"Found {len(response.results)} results")
    """
```

#### Class Documentation

```python
class EmbeddingManager:
    """Manages embedding generation with multiple provider support.
    
    The EmbeddingManager provides a unified interface for generating
    embeddings using various providers (OpenAI, Anthropic, local models).
    It handles provider selection, caching, and batch processing automatically.
    
    Attributes:
        config: Configuration for embedding providers
        cache: Optional embedding cache for performance
        
    Example:
        >>> config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
        >>> manager = EmbeddingManager(config)
        >>> embeddings = await manager.generate_embeddings(["text1", "text2"])
    """
```

## 🔍 Code Review Process

### Before Creating a Pull Request

#### Pre-submission Checklist

```bash
# Quality checks
□ All tests pass locally
□ Code follows linting standards (ruff check/format)
□ Type checking passes (mypy)
□ New tests added for new functionality
□ Documentation updated if needed
□ No secrets or credentials committed
□ Commit messages follow conventional format
□ Branch is up to date with main

# Run complete validation
make validate
```

#### PR Preparation

1. **Write clear PR description**: Explain what changes and why
2. **Reference related issues**: Link to issue numbers
3. **Include test coverage**: Show tests for new functionality
4. **Add screenshots**: For UI changes or visual improvements
5. **Document breaking changes**: Clearly mark any breaking changes

### Pull Request Requirements

#### PR Template

```markdown
## Description
Brief description of the changes and their motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed for UI changes

## Documentation
- [ ] Documentation updated
- [ ] API documentation updated if needed
- [ ] README updated if needed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is commented where necessary
- [ ] No secrets or credentials included
```

### Review Process

#### Reviewer Checklist

**Code Quality:**

- [ ] Code follows style guidelines and conventions
- [ ] Logic is clear and well-organized
- [ ] Error handling is appropriate
- [ ] Performance considerations addressed
- [ ] No obvious bugs or issues

**Testing:**

- [ ] Adequate test coverage for new functionality
- [ ] Tests are well-written and meaningful
- [ ] Edge cases and error scenarios covered
- [ ] Tests pass reliably

**Security:**

- [ ] No hardcoded secrets or credentials
- [ ] Input validation and sanitization present
- [ ] SQL injection prevention where applicable
- [ ] XSS prevention for web endpoints

**Documentation:**

- [ ] Code is well-commented where necessary
- [ ] API documentation updated
- [ ] User documentation updated if needed
- [ ] Breaking changes clearly documented

#### Review Guidelines

**For Reviewers:**

1. **Be constructive**: Focus on improving code quality
2. **Be specific**: Provide clear, actionable feedback
3. **Be timely**: Review PRs within 1-2 business days
4. **Ask questions**: Seek clarification when needed
5. **Acknowledge good work**: Recognize quality contributions

**For Contributors:**

1. **Be responsive**: Address feedback promptly
2. **Be open**: Consider alternative approaches
3. **Be thorough**: Test your changes carefully
4. **Be patient**: Allow time for thorough review
5. **Be collaborative**: Work with reviewers to improve code

### Merge Process

#### Merge Requirements

- [ ] All required reviews approved
- [ ] All CI checks passing
- [ ] No merge conflicts
- [ ] Branch is up to date with main

#### Merge Strategy

- **Squash and merge**: Preferred for feature branches
- **Merge commit**: For important milestones
- **Rebase and merge**: For clean linear history when appropriate

## 🏗️ Architecture Guidelines

### Service Layer Design

#### Base Service Pattern

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

class BaseService(ABC):
    """Base class for all services."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize service resources."""
        if self._initialized:
            return
        
        await self._setup()
        self._initialized = True
        self.logger.info(f"{self.__class__.__name__} initialized")
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        if not self._initialized:
            return
        
        await self._teardown()
        self._initialized = False
        self.logger.info(f"{self.__class__.__name__} cleaned up")
    
    @abstractmethod
    async def _setup(self) -> None:
        """Implement service-specific setup."""
        pass
    
    @abstractmethod
    async def _teardown(self) -> None:
        """Implement service-specific cleanup."""
        pass
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
```

#### Configuration Management

```python
from pydantic import BaseModel, Field
from typing import Optional

class ServiceConfig(BaseModel):
    """Base configuration for services."""
    
    name: str = Field(..., description="Service name")
    enabled: bool = Field(True, description="Service enabled flag")
    timeout: float = Field(30.0, ge=1.0, description="Service timeout")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    
    class Config:
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment

class SearchConfig(ServiceConfig):
    """Configuration for search service."""
    
    collection_name: str = Field("documents", description="Default collection")
    default_limit: int = Field(20, ge=1, le=100, description="Default result limit")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum score")
```

### Error Handling

#### Exception Hierarchy

```python
class ServiceError(Exception):
    """Base exception for service errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

class ValidationError(ServiceError):
    """Raised when input validation fails."""
    pass

class ConfigurationError(ServiceError):
    """Raised when configuration is invalid."""
    pass

class ExternalServiceError(ServiceError):
    """Raised when external service calls fail."""
    pass
```

#### Error Handling Pattern

```python
async def service_method(self, input_data: InputModel) -> OutputModel:
    """Example service method with proper error handling."""
    try:
        # Validate input
        validated_input = InputModel.model_validate(input_data)
        
        # Process request
        result = await self._process_data(validated_input)
        
        # Return response
        return OutputModel(data=result, success=True)
        
    except ValidationError as e:
        self.logger.warning(f"Validation error: {e}")
        raise
    except ExternalServiceError as e:
        self.logger.error(f"External service error: {e}")
        # Implement retry logic or fallback
        raise
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
        raise ServiceError("Internal service error") from e
```

## 🚀 Release Process

### Version Management

#### Semantic Versioning

```bash
# Format: MAJOR.MINOR.PATCH
# MAJOR: Breaking changes
# MINOR: New features (backward compatible)
# PATCH: Bug fixes

# Examples:
v1.0.0 -> v1.0.1  # Bug fix
v1.0.1 -> v1.1.0  # New feature  
v1.1.0 -> v2.0.0  # Breaking change
```

#### Version Updates

```bash
# Update version in pyproject.toml
[tool.poetry]
version = "1.1.0"

# Tag release
git tag -a v1.1.0 -m "Release v1.1.0: Add hybrid search"
git push origin v1.1.0
```

### Release Notes Template

```markdown
## v1.1.0 - 2024-01-15

### 🎉 New Features
- Add hybrid vector search with reranking
- Implement comprehensive security validation  
- Add 500+ unit tests with 90%+ coverage

### 🐛 Bug Fixes
- Fix memory leak in embedding cache
- Resolve configuration validation edge cases
- Improve error handling in web scraping

### 📚 Documentation
- Update API reference with new models
- Add comprehensive testing documentation
- Improve contributing guidelines

### ⚠️ Breaking Changes
- Configuration structure changed (migration guide available)
- API endpoint `/v1/search` now requires authentication

### 🏗️ Internal Changes
- Refactor service layer architecture
- Improve error handling patterns
- Optimize database query performance

### 📊 Performance
- 40% improvement in search latency
- 60% reduction in memory usage
- 2x faster embedding generation
```

## 🎯 Contribution Best Practices

### Getting Started Tips

1. **Start small**: Pick up "good first issue" labels
2. **Ask questions**: Don't hesitate to ask for clarification
3. **Read existing code**: Understand patterns before adding new code
4. **Follow conventions**: Match existing code style and patterns
5. **Test thoroughly**: Add comprehensive tests for your changes

### Code Quality Tips

1. **Write tests first**: TDD approach leads to better design
2. **Keep it simple**: Prefer straightforward solutions
3. **Document decisions**: Explain why, not just what
4. **Handle errors**: Consider all failure modes
5. **Think about users**: Consider the developer experience

### Collaboration Tips

1. **Communicate early**: Discuss approach before implementation
2. **Share progress**: Update issues with progress
3. **Review others**: Participate in code reviews
4. **Be patient**: Allow time for thorough review
5. **Help newcomers**: Share knowledge with new contributors

## 🆘 Getting Help

### Resources

- **[GitHub Issues](https://github.com/project/issues)**: Bug reports and feature requests
- **[GitHub Discussions](https://github.com/project/discussions)**: Questions and community help
- **[Documentation](./README.md)**: Technical documentation
- **[Architecture Guide](./architecture.md)**: System design information

### Community Guidelines

1. **Be respectful**: Treat all community members with respect
2. **Be helpful**: Share knowledge and assist others
3. **Be constructive**: Provide actionable feedback
4. **Be patient**: Allow time for responses
5. **Be inclusive**: Welcome contributors of all skill levels

### Issue Reporting

When reporting bugs or requesting features:

1. **Search existing issues** first
2. **Use issue templates** when available
3. **Provide reproduction steps** for bugs
4. **Include relevant details** (OS, Python version, etc.)
5. **Be clear and specific** about the problem or request

---

*🤝 Thank you for contributing to the AI Documentation Vector DB project! Your contributions help make this system better for everyone. Whether you're fixing bugs, adding features, improving documentation, or helping other contributors, every contribution is valuable.*
