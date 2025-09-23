# Contributing Guide

## Workflow

1. Fork the repository and clone your fork
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make changes with tests
4. Commit using conventional commits format
5. Push branch: `git push origin feature/your-feature-name`
6. Open a pull request

## Git Commit Format

We follow conventional commits specification:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, missing semicolons, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or modifying tests
- `chore:` - Maintenance tasks

Format: `type(scope): description`
Example: `feat(auth): add jwt token validation`

## Code Quality Requirements

- Run linting before commits: `npm run lint`
- All tests must pass: `npm test`
- Write unit tests for new functionality
- Maintain existing code coverage levels
- Follow established code style in the project

## Pull Request Requirements

- PR title follows conventional commits format
- Description explains changes and motivation
- Include screenshots for UI changes
- Link related issues
- Pass all CI checks

### PR Checklist

- [ ] Code follows project style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Self-reviewed changes
- [ ] No merge conflicts

## Review Process

- Maintainers review PRs within 2 business days
- Address all review comments
- CI must pass before merging
- Squash and merge for clean history
- Delete branch after merge