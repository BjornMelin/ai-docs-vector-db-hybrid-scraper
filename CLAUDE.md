# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

> **Inheritance**
> This repo inherits the global Simplicity Charter (`~/.claude/CLAUDE.md`),  
> which defines all coding, linting, testing, security and library choices  
> (uv, ruff, Pydantic v2, ≥ 80% pytest-cov, secrets in .env, etc.).
> Google format docstrings for Python code.
> Project-specific implementation context is maintained in:  
> • `TODO.md` – comprehensive task list with V1/V2 implementation roadmap  
> • `TODO-V2.md` – future feature roadmap post-V1 release  
> Always read these before coding.

## Commands
- Test: `uv run pytest --cov=src`
- Lint: `ruff check . --fix && ruff format .`  
- Services: `./scripts/start-services.sh`
- Health: `curl localhost:6333/health`

## Workflow
- Use parallel tool calling for file reads and quality checks
- Say "Create plan but DO NOT code until approved" for complex tasks
- Write tests first, run quality checks before commits

## Extended Thinking
Use these commands to trigger progressively deeper analysis:
- `think` – basic extended thinking for evaluating alternatives
- `think hard` – increased thinking budget for complex problems  
- `think harder` – higher computation time for thorough evaluation
- `ultrathink` – maximum thinking budget for the most complex scenarios
Each level allocates progressively more thinking budget for Claude to use.
