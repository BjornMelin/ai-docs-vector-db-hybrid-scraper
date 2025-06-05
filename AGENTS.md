# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

> **Inheritance**
> This repo inherits the global Simplicity Charter (`~/.claude/CLAUDE.md`),  
> which defines all coding, linting, testing, security and library choices  
> (uv, ruff, Pydantic v2, ≥ 80% pytest-cov, secrets in .env, etc.).
> Google format docstrings for Python code.
> Project-specific implementation context is maintained in:  
> • `TODO.md` - comprehensive task list with V1/V2 implementation roadmap  
> • `TODO-V2.md` - future feature roadmap post-V1 release  
> Always read these before coding.

## Commands

**Note:** Prefix all Python and test commands with `uv` since we're in a uv-managed environment.

- Test: `uv run pytest --cov=src`
- Lint: `ruff check . --fix && ruff format .`  
- Services: `./scripts/start-services.sh`
- Health: `curl localhost:6333/health`

## Workflow

### TDD Process

1. **Write tests** based on expected input/output pairs, commit
2. **Run tests** to confirm they fail (no implementation yet)
3. **Write code** to pass tests without modifying tests
4. **Iterate** until all tests pass, then commit
5. **Typecheck** when done with code changes

### Tool Usage

- **Parallel execution:** Invoke multiple independent tools simultaneously, not sequentially
- **Thinking:** Reflect on tool results quality before proceeding; use thinking to plan next steps
- **Testing:** Prefer running single tests over full test suite for performance
- **Use checklists** for complex workflows - create Markdown files to track progress on multi-step tasks

### Frontend Development Instructions

When implementing frontend/visual components:

- Don't hold back. Give it your all.
- Include all relevant user interactions and interface elements
- Add hover states, smooth transitions, and thoughtful micro-interactions
- Apply visual design principles: establish clear hierarchy, use effective contrast, maintain visual balance, and create purposeful movement
- Prioritize user experience through polished details and responsive behavior

## Extended Thinking

Use these commands to trigger progressively deeper analysis:

- `think` - basic extended thinking for evaluating alternatives
- `think hard` - increased thinking budget for complex problems
- `think harder` - higher computation time for thorough evaluation
- `ultrathink` - maximum thinking budget for the most complex scenarios

Each level allocates progressively more thinking budget for Claude to use.

## File Management

- **Avoid temporary files:** Don't create temporary scripts or helper files for iteration
- **Clean up:** If temporary files are created, remove them at task completion
- **Use exact paths:** Reference files like `src/services/embeddings/manager.py`

## Guidelines

- **Conventional commits:** Use conventional-commits format for branch names, commits, pull requests, and GitHub issues
- **Targets:** Set measurable goals like "reduce latency to <100ms for 95th percentile"
