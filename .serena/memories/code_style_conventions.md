# Code Style & Conventions

## Core Principles
- **KISS**: Prefer straightforward solutions over clever abstractions
- **YAGNI**: Implement only what's explicitly needed
- **DRY**: No duplicated logic; factor into clear helpers
- **Maintainability**: Write code you'd want to maintain six months from now

## Python Standards
- **Docstrings**: Google format for all functions/classes
- **Type Hints**: Full type annotations required
- **Validation**: Pydantic v2 models for data validation
- **Import Organization**: Standard library, third-party, local imports

## Formatting & Linting
- **Formatter**: ruff format
- **Linter**: ruff check with auto-fix
- **Type Checker**: mypy with strict configuration
- **Line Length**: 88 characters (Black compatible)

## Testing Requirements
- **Coverage**: ≥ 80% pytest coverage
- **Test Organization**: Unit, Integration, E2E structure
- **Async Testing**: pytest-asyncio with respx for HTTP mocking
- **Property Testing**: hypothesis for edge case discovery

## Security & Environment
- **Secrets**: Store in .env files, never commit
- **Configuration**: Pydantic Settings for environment management
- **Validation**: Input validation at all boundaries

## CLI Development
- **Framework**: Click with Rich for enhanced UI
- **Error Handling**: Specific exceptions, not bare Exception
- **Progress**: Rich progress indicators for long operations
- **Confirmations**: Interactive prompts for destructive operations