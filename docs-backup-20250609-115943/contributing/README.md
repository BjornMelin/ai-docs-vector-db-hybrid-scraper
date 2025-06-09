# Contributing

> **Purpose**: Guidelines for contributors and developers  
> **Audience**: Open source contributors and team members

## Contributing Documentation

### Development Setup
- [**Development Setup**](./development-setup.md) - Local development environment configuration
- [**Testing Guide**](../contributing/testing-guide.md) - Testing standards and procedures
- [**Testing Quality**](../contributing/testing-quality.md) - Quality assurance and best practices

### Architecture & Design
- [**Architecture Guide**](../contributing/architecture-guide.md) - System design principles and patterns
- [**Browser Manager Refactor**](../contributing/browser-manager-refactor.md) - Unified browser management design
- [**V1 Implementation Plan**](../contributing/v1-implementation-plan.md) - V1 development roadmap

## Getting Started

### Prerequisites
- Python 3.13+
- Docker and Docker Compose
- Git and GitHub account
- Basic understanding of vector databases

### Development Workflow

1. **Setup**: Follow [Development Setup](./development-setup.md)
2. **Branch**: Create feature branch from `main`
3. **Develop**: Implement changes with tests
4. **Test**: Run full test suite with coverage
5. **Document**: Update relevant documentation
6. **Submit**: Create pull request with clear description

## Contribution Guidelines

### Code Standards

**Python Code**:
- Use `uv` for dependency management
- Follow PEP 8 with `ruff` formatting
- Type hints required for all functions
- Pydantic models for data validation
- Google-style docstrings

**Testing Requirements**:
- Minimum 80% test coverage
- Unit tests for all new functions
- Integration tests for new features
- Performance benchmarks for critical paths

**Documentation**:
- Update relevant docs for all changes
- Follow kebab-case naming convention
- Include code examples in guides
- Add entries to appropriate README files

### Pull Request Process

1. **Description**: Clear summary of changes and motivation
2. **Testing**: Include test results and coverage report
3. **Documentation**: Update all affected documentation
4. **Review**: Address all reviewer feedback
5. **Merge**: Squash commits for clean history

## Development Commands

```bash
# Setup development environment
uv sync --dev

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Lint and format code
ruff check . --fix && ruff format .

# Type checking
uv run mypy src/

# Start development services
./scripts/start-services.sh
```

## Project Structure

Understanding the codebase organization:
- `src/` - Main application code
- `tests/` - Test suite with unit and integration tests
- `docs/` - Documentation following Diátaxis framework
- `scripts/` - Development and deployment scripts
- `config/` - Configuration templates and examples

## Related Documentation

- 🧠 [Architecture Concepts](../concepts/architecture/) - System design understanding
- 🛠️ [How-to Guides](../how-to-guides/) - Implementation examples
- 📋 [Reference](../reference/) - Technical specifications