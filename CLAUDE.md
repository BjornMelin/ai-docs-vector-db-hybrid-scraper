# CLAUDE.md

> **Inheritance**
> This repo inherits the global Simplicity Charter (`~/.claude/CLAUDE.md`),  
> which defines all coding, linting, testing, security and library choices  
> (uv, ruff, Pydantic v2, ≥ 80% pytest-cov, secrets in .env, etc.).
> Google format docstrings for Python code.

## Core Project Commands

**Note:** Prefix all Python and test commands with `uv` since we're in a uv-managed environment.

- Test: `uv run pytest --cov=src`
- Benchmark: `uv run pytest tests/benchmarks/ --benchmark-only`
- Lint: `ruff check . --fix && ruff format .`

### Testing Quality Standards

**Testing Anti-Patterns to AVOID:**

❌ **Coverage-Driven Testing**: Never create tests solely to hit coverage metrics or line numbers  
❌ **Implementation Detail Testing**: Don't test private methods or internal implementation details  
❌ **Heavy Internal Mocking**: Avoid over-mocking internal components; mock at boundaries  
❌ **Shared Mutable State**: Tests that depend on execution order or share state between functions  
❌ **Magic Values**: Hardcoded test data without clear business meaning  
❌ **Timing-Dependent Tests**: Don't rely on real timers or timing for async/cache tests  
❌ **Giant Test Functions**: Tests that verify multiple behaviors in a single function

**Testing Best Practices to FOLLOW:**

✅ **Functional Organization**: Structure tests by business functionality (Unit, Integration, E2E)  
✅ **Behavior-Driven Testing**: Test observable behavior, not implementation details  
✅ **Boundary Mocking**: Mock external services (APIs, databases) not internal logic  
✅ **AAA Pattern**: Arrange, Act, Assert structure for clear test flow  
✅ **Property-Based Testing**: Use `hypothesis` for data generation and edge case discovery  
✅ **Async Test Patterns**: Proper `pytest-asyncio` usage with `respx` for HTTP mocking  
✅ **Descriptive Test Names**: Names that explain business value and expected behavior

**Test Quality Checklist:**

- [ ] Tests organized by functionality, not coverage metrics
- [ ] All scenarios represent realistic usage patterns
- [ ] `respx` used for HTTP mocking with proper setup/teardown
- [ ] Proper async/await patterns with `pytest-asyncio`
- [ ] No artificial timing dependencies or private method testing
- [ ] Test names describe business value and expected behavior
- [ ] 80%+ coverage achieved through meaningful scenarios, not line-targeting
- [ ] AI/ML operations tested for properties (dimensions, types) not exact values

### Tool Usage

- **Testing:** Prefer running single tests over full test suite for performance
- **Use checklists** for complex workflows - create Markdown files to track progress on multi-step tasks

### File Management

- **Avoid temporary files:** Don't create temporary scripts or helper files for iteration
- **Clean up:** If temporary files are created, remove them at task completion
- **Use exact paths:** Reference files like `src/services/embeddings/manager.py`

### Guidelines

- **Conventional commits:** Use conventional-commits format for branch names, commits, pull requests, and GitHub issues
- **Targets:** Set measurable goals like "reduce latency to <100ms for 95th percentile"

---

## Task Master AI - Quick Start Workflow

1. `initialize_project` → `parse_prd` → `analyze_project_complexity` → `expand_all`
2. Daily: `next_task` → `get_task` → work → `update_subtask` → `set_task_status`

### Task Status Values

- `pending`, `in-progress`, `done`, `deferred`, `cancelled`, `blocked`

### File Management

- Never manually edit `tasks.json` or `.taskmaster/config.json`
- Use `generate` to regenerate task files after manual changes

### Research Mode

Add `{research: true}` to any MCP call for enhanced AI analysis
